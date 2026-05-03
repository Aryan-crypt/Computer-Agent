"""
Configuration for Remote PC Control Agent
"""

import os
from pathlib import Path

# ============ SECURITY CONFIG ============
REQUIRE_CONFIRMATION = True  # Ask "Proceed? (y/n)" before executing
ALLOW_SCREENSHOT_COMMAND = True  # Allow /screenshot to see your screen
EMERGENCY_STOP_ENABLED = True  # Allow /stop to abort tasks

# ============ FEATURE: RATE LIMITING CONFIG ============
RATE_LIMIT_MAX_REQUESTS = 10  # Max commands allowed per user within the time window
RATE_LIMIT_WINDOW_SECONDS = 60  # Time window in seconds for rate limiting

# ============ FEATURE: FILE TRANSFER CONFIG ============
# Automatically resolves the active Windows user's Downloads folder
DOWNLOADS_FOLDER = str(Path.home() / "Downloads")

# ============ FEATURE: WEBCAM CONFIG ============
WEBCAM_INDEX = 0  # Default camera index (0 is usually the built-in laptop webcam)

# ============ FEATURE: CUSTOM SHORTCUTS CONFIG ============
ALIAS_FILE = "aliases.json"  # File to store custom aliases persistently

# ============ FEATURE: SELF-RECONNECTION CONFIG ============
RECONNECT_BASE_DELAY = 10  # Initial wait time (seconds) after internet drops
RECONNECT_MAX_DELAY = 60   # Maximum wait time (seconds) during exponential backoff

# ============ AGENT CONFIG ============
MAX_REPLAN_ATTEMPTS = 3
STEP_UPDATE_INTERVAL = 3  # Send update every N steps (0 = no updates during task)
STILL_WORKING_PING_INTERVAL = 60  # Send "still working" every N seconds

# ============ TIMING CONFIG (Fix #14) ============
ACTION_DELAY_SECONDS = 0.5  # Delay after keyboard shortcuts/actions
CLICK_DELAY_SECONDS = 0.3   # Delay after mouse clicks/scrolls
WINDOW_LOAD_DELAY_SECONDS = 1.0  # Delay to wait for windows/websites to load

# ============ TASK EXECUTION CONFIG (Fix #7) ============
TASK_TIMEOUT_SECONDS = 300  # Maximum time (in seconds) a task is allowed to run before timing out (5 minutes)

# ============ FEATURE: NOTIFICATION FORWARDING CONFIG ============
ENABLE_NOTIFICATION_FORWARDING = True  # Set to True to forward PC notifications to Telegram (Requires win32gui)

# ============ FEATURE: LIVE SCREEN STREAMING CONFIG ============
SCREEN_STREAM_DURATION = 10  # Duration of the screen recording in seconds
SCREEN_STREAM_FPS = 10.0     # Frames per second for the screen recording (lower = smaller file size)

# ============ FEATURE: MICROPHONE RECORDING CONFIG ============
MIC_RECORD_DURATION = 10     # Duration of the microphone recording in seconds
MIC_SAMPLE_RATE = 44100      # Audio sample rate (44100 is standard CD quality)

# ============ LOGGING CONFIG ============
SAVE_TASK_HISTORY = True
TASK_HISTORY_FILE = "task_history.json"