"""
Automatic Gemini API Key Rotation
=================================
Fixes: "Gemini API key quota runs out after some usage".

Add all your keys to GEMINI_API_KEYS in API.py. This manager:
  1. Uses the first healthy key.
  2. On a quota / rate-limit / invalid-key error (HTTP 429, 401, 403,
     RESOURCE_EXHAUSTED, "API key not valid"...), the key is benched and
     the SAME request is instantly retried with the next key.
  3. Benched keys return automatically after a cooldown that doubles on
     repeated failures (60s -> 2m -> 4m -> ... -> 30m max), so per-minute
     limits recover fast while daily-exhausted keys stay benched longer.
  4. If ALL keys are benched, calls wait briefly (max ~90s) for the
     soonest key to recover before raising the error.

Usage (drop-in replacement for client.models.generate_content):
    from key_rotation import gemini_keys
    response = gemini_keys.generate_content(model="gemini-2.0-flash", contents=[...])
"""

import logging
import threading
import time
from typing import List, Optional

import google.genai as genai

try:
    from API import GEMINI_API_KEYS
except ImportError:
    # Older API.py without the key list -> fall back to the single variable
    try:
        from API import GEMINI_API_KEY as GEMINI_API_KEYS
    except ImportError:
        GEMINI_API_KEYS = []

logger = logging.getLogger(__name__)

# HTTP status codes meaning "the KEY is the problem, switch keys".
_KEY_ERROR_CODES = {401, 403, 429}

# Words inside an error message that mean the same thing.
_KEY_ERROR_KEYWORDS = (
    "quota", "resource_exhausted", "resource has been exhausted",
    "rate limit", "rate_limit", "ratelimit", "too many requests",
    "exceeded", "api key", "api_key", "apikey", "unauthenticated",
    "permission denied", "permission_denied", "unauthorized", "forbidden",
)

# Exception class names (google-genai / google.api_core) that are key-related.
_KEY_ERROR_TYPES = {"ResourceExhausted", "PermissionDenied", "Unauthenticated"}

# Placeholder text a user may accidentally leave in API.py.
_PLACEHOLDER_MARKERS = ("paste_your", "your_key", "your_api", "insert_", "placeholder")


class GeminiKeyManager:
    """Pool of Gemini API keys with automatic failover. Thread-safe."""

    BASE_COOLDOWN = 60        # seconds benched after 1st failure
    MAX_COOLDOWN = 30 * 60    # cap for repeatedly failing keys
    MAX_TOTAL_WAIT = 90       # longest a call waits when ALL keys are benched

    def __init__(self, api_keys: Optional[List[str]] = None):
        api_keys = api_keys if api_keys is not None else GEMINI_API_KEYS
        if isinstance(api_keys, str):
            api_keys = [api_keys]

        self._keys: List[str] = []
        for raw in api_keys:
            key = (raw or "").strip()
            if not key:
                continue
            if any(m in key.lower() for m in _PLACEHOLDER_MARKERS):
                logger.warning(f"Skipping placeholder key '...{key[-6:]}' - edit API.py!")
                continue
            self._keys.append(key)

        if not self._keys:
            raise RuntimeError(
                "No Gemini API keys configured! Open API.py and add keys to the "
                "GEMINI_API_KEYS list (free keys: https://aistudio.google.com/app/apikey)."
            )

        self._lock = threading.Lock()
        self._index = 0                                 # currently preferred key
        self._clients = {}                              # key -> cached genai.Client
        self._cooldown_until = [0.0] * len(self._keys)  # bench timestamps
        self._fail_streak = [0] * len(self._keys)

        logger.info(f"🔑 GeminiKeyManager initialised with {len(self._keys)} API key(s).")
        if len(self._keys) == 1:
            logger.warning(
                "Only 1 Gemini key configured - automatic failover impossible. "
                "Add more keys to GEMINI_API_KEYS in API.py!"
            )

    # ---------------- PUBLIC API ---------------- #

    def get_client(self) -> genai.Client:
        """A genai.Client for the best currently-available key."""
        with self._lock:
            return self._client_for(self._pick_locked())

    def generate_content(self, model: str, contents, **kwargs):
        """
        Drop-in replacement for client.models.generate_content(...) that
        rotates API keys and retries automatically on quota/key errors.
        """
        last_error: Optional[Exception] = None
        wait_deadline = time.time() + self.MAX_TOTAL_WAIT
        attempts_left = 2 * len(self._keys)

        while attempts_left > 0:
            with self._lock:
                idx = self._pick_locked()

            if idx is None:
                # Every key is benched -> wait a bit for one to recover.
                wait = self._seconds_until_any_available()
                if wait <= 0 or time.time() + wait > wait_deadline:
                    break
                logger.warning(
                    f"All {len(self._keys)} Gemini keys are on cooldown - "
                    f"waiting {wait:.0f}s for the next one to recover..."
                )
                time.sleep(wait)
                continue

            attempts_left -= 1
            with self._lock:
                client = self._client_for(idx)
                self._index = idx

            try:
                response = client.models.generate_content(
                    model=model, contents=contents, **kwargs
                )
            except Exception as e:
                if not self._is_key_error(e):
                    raise  # not key-related (bad request, network...) -> bubble up
                last_error = e
                with self._lock:
                    self._bench_locked(idx, e)
                continue  # retry the SAME request with the next key

            with self._lock:
                self._fail_streak[idx] = 0  # healthy again
            return response

        raise last_error if last_error else RuntimeError(
            "All Gemini API keys are benched/exhausted."
        )

    def status_report(self) -> str:
        """Human-readable status of every key (used by /keys command)."""
        with self._lock:
            now = time.time()
            lines = [f"🔑 **Gemini Keys: {len(self._keys)} configured**"]
            for i, key in enumerate(self._keys):
                tail = key[-6:]
                remaining = self._cooldown_until[i] - now
                if remaining > 0:
                    lines.append(f"{i + 1}. `…{tail}` ⏳ benched ({int(remaining)}s left)")
                elif i == self._index:
                    lines.append(f"{i + 1}. `…{tail}` 🟢 **in use**")
                else:
                    lines.append(f"{i + 1}. `…{tail}` 🟢 ready")
            return "\n".join(lines)

    # ---------------- INTERNALS (*_locked = must hold self._lock) ---------------- #

    def _client_for(self, idx: int) -> genai.Client:
        key = self._keys[idx]
        if key not in self._clients:
            self._clients[key] = genai.Client(api_key=key)
        return self._clients[key]

    def _pick_locked(self) -> Optional[int]:
        """Best available (non-benched) key index, or None if all benched."""
        now = time.time()
        if self._cooldown_until[self._index] <= now:
            return self._index
        for i, until in enumerate(self._cooldown_until):
            if until <= now:
                return i
        return None

    def _bench_locked(self, idx: int, error: Exception):
        self._fail_streak[idx] += 1
        cooldown = min(
            self.BASE_COOLDOWN * (2 ** (self._fail_streak[idx] - 1)),
            self.MAX_COOLDOWN,
        )
        self._cooldown_until[idx] = time.time() + cooldown
        logger.warning(
            f"🔁 Gemini key #{idx + 1} (…{self._keys[idx][-6:]}) benched for "
            f"{cooldown}s → {self._short(error)}"
        )
        nxt = self._pick_locked()
        if nxt is not None:
            logger.info(f"➡️ Switching to Gemini key #{nxt + 1} (…{self._keys[nxt][-6:]}).")

    def _seconds_until_any_available(self) -> float:
        with self._lock:
            now = time.time()
            return max(0.0, min(self._cooldown_until) - now)

    # ---------------- Error classification ---------------- #

    @classmethod
    def _is_key_error(cls, error: Exception) -> bool:
        """True if switching API keys could fix this error."""
        for attr in ("code", "status_code"):  # google-genai ClientError has .code
            code = getattr(error, attr, None)
            if isinstance(code, int) and code in _KEY_ERROR_CODES:
                return True
        if type(error).__name__ in _KEY_ERROR_TYPES:
            return True
        msg = str(error).lower()
        return any(word in msg for word in _KEY_ERROR_KEYWORDS)

    @staticmethod
    def _short(error: Exception) -> str:
        return " ".join(str(error).split())[:160]


# One shared instance for the whole application
gemini_keys = GeminiKeyManager()