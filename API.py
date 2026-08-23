# ==========================================
# API KEYS CONFIGURATION
# ==========================================
# HOW TO ADD MORE GEMINI KEYS (FREE):
# Each Google account gets its own free quota.
# Create a key per account at: https://aistudio.google.com/app/apikey
# then paste them below, one per line. When one key runs out of quota
# or gets blocked, the agent AUTOMATICALLY switches to the next one.

GEMINI_API_KEYS = [
    "YOUR_API_KEY",   # Account 1
    "PASTE_YOUR_SECOND_KEY_HERE",                 # Account 2
    "PASTE_YOUR_THIRD_KEY_HERE",                  # Account 3 (add as many as you like)
]

# --- Do not edit below (kept for backward compatibility) ---
GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""

OpenRouter_API_KEY = "YOUR_API_KEY"
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
AUTHORIZED_USERS = [123,123,123]