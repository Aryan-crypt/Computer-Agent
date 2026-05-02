"""
Telegram Bot Interface for Remote PC Control
Handles all communication between user and agent
"""

import os
import json
import time
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue

# Telegram library
from telegram import (
    Update, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes
)

from config import *
from main import PCControlAgent
from API import *


class TaskStatus(Enum):
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class TaskRecord:
    """Record of a completed/running task"""
    task_id: int
    command: str
    status: TaskStatus = TaskStatus.IDLE
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    steps_completed: int = 0
    total_steps: int = 0
    replans: int = 0
    result_summary: str = ""
    error: str = ""


class TaskHistory:
    """Persistent task history storage"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.tasks: Dict[int, TaskRecord] = {}
        self._next_id = 1
        self._load()
    
    def _load(self):
        """Load history from file"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    for tid, tdata in data.get("tasks", {}).items():
                        self.tasks[int(tid)] = TaskRecord(
                            task_id=int(tid),
                            command=tdata["command"],
                            status=TaskStatus(tdata["status"]),
                            started_at=tdata.get("started_at"),
                            completed_at=tdata.get("completed_at"),
                            steps_completed=tdata.get("steps_completed", 0),
                            total_steps=tdata.get("total_steps", 0),
                            replans=tdata.get("replans", 0),
                            result_summary=tdata.get("result_summary", ""),
                            error=tdata.get("error", "")
                        )
                    self._next_id = data.get("next_id", len(self.tasks) + 1)
            except Exception as e:
                print(f"Error loading task history: {e}")
    
    def _save(self):
        """Save history to file"""
        if not SAVE_TASK_HISTORY:
            return
        try:
            data = {
                "next_id": self._next_id,
                "tasks": {}
            }
            for tid, task in self.tasks.items():
                data["tasks"][str(tid)] = {
                    "command": task.command,
                    "status": task.status.value,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                    "steps_completed": task.steps_completed,
                    "total_steps": task.total_steps,
                    "replans": task.replans,
                    "result_summary": task.result_summary,
                    "error": task.error
                }
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving task history: {e}")
    
    def create_task(self, command: str) -> TaskRecord:
        """Create a new task record"""
        task = TaskRecord(
            task_id=self._next_id,
            command=command,
            status=TaskStatus.QUEUED
        )
        self.tasks[self._next_id] = task
        self._next_id += 1
        self._save()
        return task
    
    def update_task(self, task: TaskRecord):
        """Update existing task record"""
        self.tasks[task.task_id] = task
        self._save()
    
    def get_recent(self, count: int = 10) -> list:
        """Get recent tasks"""
        return sorted(
            self.tasks.values(), 
            key=lambda t: t.task_id, 
            reverse=True
        )[:count]


class TelegramPCInterface:
    """Main Telegram Bot interface for PC Control"""
    
    def __init__(self):
        self.pc_agent = PCControlAgent()
        self.task_history = TaskHistory(TASK_HISTORY_FILE)
        
        # Task management
        self.current_task: Optional[TaskRecord] = None
        self.task_queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.is_busy = False
        
        # Thread management
        self.worker_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Pending confirmations
        self.pending_commands: Dict[int, str] = {}  # chat_id -> command
        
        # Main event loop reference (for thread-safe messaging)
        self.main_loop = None
        
        # Build application
        self.application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        self._register_handlers()
    
    def _register_handlers(self):
        """Register all command and message handlers"""
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("task", self.cmd_task))
        self.application.add_handler(CommandHandler("status", self.cmd_status))
        self.application.add_handler(CommandHandler("screenshot", self.cmd_screenshot))
        self.application.add_handler(CommandHandler("stop", self.cmd_stop))
        self.application.add_handler(CommandHandler("history", self.cmd_history))
        self.application.add_handler(CommandHandler("clear", self.cmd_clear))
        
        # Handle confirmation button presses
        self.application.add_handler(CallbackQueryHandler(self.callback_handler))
        
        # Handle plain text as task commands
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, 
            self.handle_text_message
        ))
    
    # ============ AUTHORIZATION CHECK ============
    
    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized to use the bot"""
        return user_id in AUTHORIZED_USERS
    
    async def _check_auth(self, update: Update) -> bool:
        """Check authorization and send error if not"""
        # CRITICAL: Capture the main event loop for thread-safe calls from ANY entry point
        if not self.main_loop:
            self.main_loop = asyncio.get_running_loop()
        
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text(
                "⛔ **UNAUTHORIZED**\n\n"
                "You are not authorized to use this bot.\n"
                f"Your ID: `{update.effective_user.id}`",
                parse_mode="Markdown"
            )
            return False
        return True
    
    # ============ COMMAND HANDLERS ============
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        if not await self._check_auth(update):
            return
        
        # CRITICAL: Capture the main event loop for thread-safe calls
        if not self.main_loop:
            self.main_loop = asyncio.get_running_loop()
        
        await update.message.reply_text(
            "🤖 **PC Control Agent - Remote Interface**\n\n"
            "I can control your home PC remotely!\n\n"
            "**Quick Commands:**\n"
            "• Send any text → I'll do that task\n"
            "• `/status` → See what I'm doing\n"
            "• `/screenshot` → See your screen\n"
            "• `/stop` → Emergency stop\n"
            "• `/history` → Past tasks\n"
            "• `/help` → Full help\n\n"
            "🔒 *Secured by Telegram ID whitelist*",
            parse_mode="Markdown"
        )
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        if not await self._check_auth(update):
            return
        
        help_text = """
📖 **FULL HELP GUIDE**

**Sending Tasks:**
• Just type what you want: `open notepad and write hello`
• Or use: `/task open calculator and do 5+3`

**Monitoring:**
• `/status` - Current task status
• `/screenshot` - See your screen right now
• `/history` - See past 10 tasks

**Emergency:**
• `/stop` - Immediately stop current task

**Tips:**
• Tasks run sequentially (one at a time)
• You'll be notified when task completes
• If offline, messages queue until you're back
• Agent auto-replans on failures (max 3x)

**Security:**
• Only whitelisted Telegram IDs can command
• Confirmation required before execution
• Emergency stop always available
"""
        await update.message.reply_text(help_text, parse_mode="Markdown")
    
    async def cmd_task(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /task command"""
        if not await self._check_auth(update):
            return
        
        if not context.args:
            await update.message.reply_text(
                "❌ Please provide a task description.\n\n"
                "Usage: `/task <your task here>`\n"
                "Example: `/task open notepad and write hello world`",
                parse_mode="Markdown"
            )
            return
        
        command = " ".join(context.args)
        await self._process_command(command, update)
    
    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle plain text messages as task commands"""
        if not await self._check_auth(update):
            return
        
        # Ignore if it's a pending confirmation (handled by callback)
        if update.effective_chat.id in self.pending_commands:
            return
        
        command = update.message.text.strip()
        
        # Ignore very short messages (likely accidental)
        if len(command) < 3:
            return
        
        await self._process_command(command, update)
    
    async def _process_command(self, command: str, update: Update):
        """Process a task command with optional confirmation"""
        chat_id = update.effective_chat.id
        
        with self.lock:
            if self.is_busy:
                await update.message.reply_text(
                    f"⏳ **Busy right now**\n\n"
                    f"Currently working on:\n"
                    f"• Task #{self.current_task.task_id}: `{self.current_task.command[:50]}...`\n\n"
                    f"Your task has been queued:\n"
                    f"• `{command[:50]}{'...' if len(command) > 50 else ''}`\n\n"
                    f"Queue size: {self.task_queue.qsize()}",
                    parse_mode="Markdown"
                )
                self.task_queue.put((command, chat_id))
                return
        
        if REQUIRE_CONFIRMATION:
            # Show confirmation buttons
            keyboard = [
                [
                    InlineKeyboardButton("✅ Execute", callback_data="confirm_yes"),
                    InlineKeyboardButton("❌ Cancel", callback_data="confirm_no")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            self.pending_commands[chat_id] = command
            
            # Truncate long commands for display
            display_cmd = command[:100] + "..." if len(command) > 100 else command
            
            await update.message.reply_text(
                f"📋 **Task Preview:**\n\n"
                f"`{display_cmd}`\n\n"
                f"⚠️ This will execute on your home PC.\n"
                f"Proceed?",
                parse_mode="Markdown",
                reply_markup=reply_markup
            )
        else:
            # Execute directly without confirmation
            await self._start_task(command, chat_id)
    
    async def callback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button presses"""
        query = update.callback_query
        await query.answer()
        
        chat_id = query.message.chat_id
        
        if query.data == "confirm_yes":
            command = self.pending_commands.pop(chat_id, None)
            if command:
                await query.edit_message_text(
                    f"✅ **Confirmed!**\n\nStarting: `{command[:50]}...`",
                    parse_mode="Markdown"
                )
                await self._start_task(command, chat_id)
            else:
                await query.edit_message_text("❌ Command expired. Please try again.")
        
        elif query.data == "confirm_no":
            command = self.pending_commands.pop(chat_id, None)
            await query.edit_message_text("❌ Task cancelled.")
    
    async def _start_task(self, command: str, chat_id: int):
        """Start executing a task"""
        with self.lock:
            self.is_busy = True
            self.stop_flag.clear()
            
            # Create task record
            task = self.task_history.create_task(command)
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()
            self.task_history.update_task(task)
            self.current_task = task
        
        # Start execution in background thread
        self.worker_thread = threading.Thread(
            target=self._execute_task_thread,
            args=(command, chat_id, task.task_id),
            daemon=True
        )
        self.worker_thread.start()
    
    def _execute_task_thread(self, command: str, chat_id: int, task_id: int):
        """Execute task in background thread"""
        
        try:
            # Execute the task
            results = self.pc_agent.execute_task(command)
            
            # Check if stopped
            if self.stop_flag.is_set():
                self.current_task.status = TaskStatus.STOPPED
                self.current_task.completed_at = datetime.now().isoformat()
                self.current_task.result_summary = "Stopped by user"
                self.task_history.update_task(self.current_task)
                
                # Send stop notification safely
                self._send_safe(self._send_message(
                    chat_id,
                    "🛑 **TASK STOPPED**\n\n"
                    f"Task #{task_id} was stopped by user.",
                    parse_mode="Markdown"
                ))
            else:
                # Update task record
                self.current_task.status = TaskStatus.COMPLETED if results["success"] else TaskStatus.FAILED
                self.current_task.completed_at = datetime.now().isoformat()
                self.current_task.steps_completed = results["steps_completed"]
                self.current_task.total_steps = results["total_steps"]
                self.current_task.replans = results.get("replans", 0)
                self.current_task.result_summary = self._summarize_results(results)
                self.task_history.update_task(self.current_task)
                
                # Send completion notification safely
                self._send_safe(self._send_completion_message(chat_id, task_id, results))
        
        except Exception as e:
            self.current_task.status = TaskStatus.FAILED
            self.current_task.completed_at = datetime.now().isoformat()
            self.current_task.error = str(e)
            self.task_history.update_task(self.current_task)
            
            self._send_safe(self._send_message(
                chat_id,
                f"❌ **TASK ERROR**\n\n"
                f"Task #{task_id} failed with error:\n"
                f"`{str(e)[:500]}`",
                parse_mode="Markdown"
            ))
        
        finally:
            with self.lock:
                self.is_busy = False
                self.current_task = None
            
            # Process next task in queue
            self._process_queue()
    
    def _summarize_results(self, results: Dict) -> str:
        """Create text summary of results"""
        summary_parts = []
        
        if results.get("step_results"):
            for sr in results["step_results"]:
                status = "✓" if sr["success"] else "✗"
                summary_parts.append(f"{status} {sr['description']}")
        
        return "\n".join(summary_parts[:20])  # Limit to 20 steps
    
    async def _send_completion_message(self, chat_id: int, task_id: int, results: Dict):
        """Send task completion message with screenshot"""
        
        status_emoji = "✅" if results["success"] else "❌"
        status_text = "COMPLETED" if results["success"] else "FAILED/ABORTED"
        
        # Build message
        message = (
            f"{status_emoji} **TASK {status_text}**\n\n"
            f"📋 Task #{task_id}\n"
            f"📊 Steps: {results['steps_completed']}/{results['total_steps']}\n"
            f"🔄 Re-plans: {results.get('replans', 0)}\n"
        )
        
        # Add step details (truncated)
        if results.get("step_results"):
            message += "\n**Steps:**\n"
            for sr in results["step_results"][-10:]:  # Show last 10 steps
                status = "✓" if sr["success"] else "✗"
                message += f"  {status} {sr['description']}\n"
            
            if len(results["step_results"]) > 10:
                message += f"  ... and {len(results['step_results']) - 10} more steps\n"
        
        # Send text message
        await self._send_message(chat_id, message, parse_mode="Markdown")
        
        # Send final screenshot
        try:
            screenshot = self.pc_agent.take_screenshot("final_result.png")
            with open("final_result.png", "rb") as photo:
                await self._send_photo(chat_id, photo, caption="📸 Final screen state:")
            self.pc_agent.cleanup_screenshot("final_result.png")
        except Exception as e:
            await self._send_message(chat_id, f"⚠️ Could not send screenshot: {e}")
    
    async def _send_message(self, chat_id: int, text: str, parse_mode: str = None):
        """Thread-safe message sending"""
        try:
            await self.application.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode
            )
        except Exception as e:
            print(f"Error sending message: {e}")
    
    async def _send_photo(self, chat_id: int, photo, caption: str = None):
        """Thread-safe photo sending"""
        try:
            await self.application.bot.send_photo(
                chat_id=chat_id,
                photo=photo,
                caption=caption
            )
        except Exception as e:
            print(f"Error sending photo: {e}")

    def _send_safe(self, coro):
        """Safely run async code from a background thread"""
        if self.main_loop and not self.main_loop.is_closed():
            asyncio.run_coroutine_threadsafe(coro, self.main_loop)
        else:
            print("Warning: Event loop not available, cannot send Telegram message.")
    
    def _process_queue(self):
        """Process next task in queue if any"""
        try:
            if not self.task_queue.empty():
                command, chat_id = self.task_queue.get_nowait()
                self._send_safe(self._start_task(command, chat_id))
        except queue.Empty:
            pass
    
    # ============ MONITORING COMMANDS ============
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        if not await self._check_auth(update):
            return
        
        with self.lock:
            if self.is_busy and self.current_task:
                elapsed = ""
                if self.current_task.started_at:
                    start = datetime.fromisoformat(self.current_task.started_at)
                    elapsed = str(datetime.now() - start).split('.')[0]
                
                status_msg = (
                    f"⏳ **CURRENTLY WORKING**\n\n"
                    f"📋 Task #{self.current_task.task_id}\n"
                    f"📝 `{self.current_task.command[:100]}...`\n"
                    f"⏱️ Elapsed: {elapsed}\n"
                    f"📊 Status: {self.current_task.status.value}\n"
                    f"📥 Queue: {self.task_queue.qsize()} tasks waiting"
                )
            else:
                status_msg = (
                    f"😴 **IDLE**\n\n"
                    f"PC Agent is ready for commands.\n"
                    f"📥 Queue: {self.task_queue.qsize()} tasks waiting"
                )
        
        await update.message.reply_text(status_msg, parse_mode="Markdown")
    
    async def cmd_screenshot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /screenshot command - send current screen"""
        if not await self._check_auth(update):
            return
        
        if not ALLOW_SCREENSHOT_COMMAND:
            await update.message.reply_text("⛔ Screenshot command is disabled.")
            return
        
        await update.message.reply_text("📸 Taking screenshot...")
        
        try:
            screenshot_path = "remote_screenshot.png"
            self.pc_agent.take_screenshot(screenshot_path)
            
            with open(screenshot_path, "rb") as photo:
                await update.message.reply_photo(
                    photo=photo,
                    caption=f"📸 Current screen state\n"
                           f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
            
            self.pc_agent.cleanup_screenshot(screenshot_path)
        
        except Exception as e:
            await update.message.reply_text(f"❌ Failed to take screenshot: {e}")
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command - emergency stop"""
        if not await self._check_auth(update):
            return
        
        if not EMERGENCY_STOP_ENABLED:
            await update.message.reply_text("⛔ Stop command is disabled.")
            return
        
        with self.lock:
            if self.is_busy:
                self.stop_flag.set()
                # Move mouse to corner to trigger pyautogui failsafe
                import pyautogui
                pyautogui.moveTo(0, 0)
                
                await update.message.reply_text(
                    "🛑 **EMERGENCY STOP ACTIVATED**\n\n"
                    "Moving mouse to failsafe corner...\n"
                    "Task will stop shortly.",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text("😊 Nothing is running right now.")
        
        # Clear queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
    
    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history command"""
        if not await self._check_auth(update):
            return
        
        recent = self.task_history.get_recent(10)
        
        if not recent:
            await update.message.reply_text("📭 No task history yet.")
            return
        
        message = "📋 **Recent Tasks:**\n\n"
        
        for task in recent:
            status_emoji = {
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.STOPPED: "🛑",
                TaskStatus.RUNNING: "⏳",
                TaskStatus.QUEUED: "📥"
            }.get(task.status, "❓")
            
            cmd_short = task.command[:40] + "..." if len(task.command) > 40 else task.command
            
            message += f"{status_emoji} `#{task.task_id}` {cmd_short}\n"
            
            if task.completed_at:
                message += f"   Steps: {task.steps_completed}/{task.total_steps}"
                if task.replans > 0:
                    message += f" | Re-plans: {task.replans}"
                message += "\n"
        
        await update.message.reply_text(message, parse_mode="Markdown")
    
    async def cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command - clear task history"""
        if not await self._check_auth(update):
            return
        
        self.task_history.tasks = {}
        self.task_history._next_id = 1
        self.task_history._save()
        
        await update.message.reply_text("🗑️ Task history cleared.")
    
    # ============ RUN METHODS ============
    
    def run(self):
        """Start the bot (blocking)"""
        print("=" * 60)
        print("PC CONTROL AGENT - TELEGRAM INTERFACE")
        print("=" * 60)
        print(f"Authorized users: {AUTHORIZED_USERS}")
        print(f"Confirmation required: {REQUIRE_CONFIRMATION}")
        print(f"Emergency stop: {EMERGENCY_STOP_ENABLED}")
        print("=" * 60)
        print("Bot is running. Send commands via Telegram.")
        print("Press Ctrl+C to stop.")
        print("=" * 60)
        
        self.application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
    
    async def run_async(self):
        """Start the bot (async)"""
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )


def main():
    """Entry point"""
    bot = TelegramPCInterface()
    bot.run()


if __name__ == "__main__":
    main()