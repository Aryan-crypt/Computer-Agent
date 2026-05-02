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

# ============ LOGGING CONFIG ============
SAVE_TASK_HISTORY = True
TASK_HISTORY_FILE = "task_history.json"