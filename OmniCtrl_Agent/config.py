"""
Configuration for Remote PC Control Agent
"""

# ============ SECURITY CONFIG ============
REQUIRE_CONFIRMATION = True  # Ask "Proceed? (y/n)" before executing
ALLOW_SCREENSHOT_COMMAND = True  # Allow /screenshot to see your screen
EMERGENCY_STOP_ENABLED = True  # Allow /stop to abort tasks

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

# ============ LOGGING CONFIG ============
SAVE_TASK_HISTORY = True
TASK_HISTORY_FILE = "task_history.json"