"""
Telegram Bot Interface for Remote PC Control
Handles all communication between user and agent
Enhanced with: Self-Reconnect, Rate Limiting, Voice Commands, Scheduled Tasks,
               Aliases, File Transfer, System Monitoring, Clipboard Sync, Webcam Capture.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import requests
import pyperclip
import psutil
import cv2
from pathlib import Path
from collections import defaultdict

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
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Google Genai for Voice STT
import google.genai as genai
from google.genai import types as genai_types

from OmniCtrl_Agent.config import *
from main import PCControlAgent
from API import *

# Fallback for timeout config in case it's missing before config.py is updated
try:
    TASK_TIMEOUT_SECONDS
except NameError:
    TASK_TIMEOUT_SECONDS = 300

# Fallback for new configs
try:
    DOWNLOADS_FOLDER
except NameError:
    DOWNLOADS_FOLDER = str(Path.home() / "Downloads")

# Set up logging (Fix #15)
logger = logging.getLogger(__name__)


# ==========================================
# FEATURE: Rate Limiter
# ==========================================
class RateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.requests = defaultdict(list)

    def is_allowed(self, user_id: int) -> bool:
        now = datetime.now()
        self.requests[user_id] = [t for t in self.requests[user_id] if now - t < self.window]
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        self.requests[user_id].append(now)
        return True


# ==========================================
# FEATURE: Custom Shortcuts (Aliases)
# ==========================================
class AliasManager:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.aliases = self._load()

    def _load(self) -> Dict[str, str]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading aliases: {e}")
        return {}

    def _save(self):
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.aliases, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving aliases: {e}")

    def get(self, key: str) -> Optional[str]:
        return self.aliases.get(key.lower().strip())

    def add(self, short: str, full: str) -> bool:
        if len(short.split()) > 1: return False # Aliases must be single word/char
        self.aliases[short.lower().strip()] = full.strip()
        self._save()
        return True

    def remove(self, short: str) -> bool:
        if short.lower().strip() in self.aliases:
            del self.aliases[short.lower().strip()]
            self._save()
            return True
        return False

    def list_all(self) -> Dict[str, str]:
        return self.aliases


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
                logger.error(f"Error loading task history: {e}")
    
    def _save(self):
        if not SAVE_TASK_HISTORY:
            return
        try:
            data = {"next_id": self._next_id, "tasks": {}}
            for tid, task in self.tasks.items():
                data["tasks"][str(tid)] = {
                    "command": task.command, "status": task.status.value,
                    "started_at": task.started_at, "completed_at": task.completed_at,
                    "steps_completed": task.steps_completed, "total_steps": task.total_steps,
                    "replans": task.replans, "result_summary": task.result_summary, "error": task.error
                }
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving task history: {e}")
    
    def create_task(self, command: str) -> TaskRecord:
        task = TaskRecord(task_id=self._next_id, command=command, status=TaskStatus.QUEUED)
        self.tasks[self._next_id] = task
        self._next_id += 1
        self._save()
        return task
    
    def update_task(self, task: TaskRecord):
        self.tasks[task.task_id] = task
        self._save()
    
    def get_recent(self, count: int = 10) -> list:
        return sorted(self.tasks.values(), key=lambda t: t.task_id, reverse=True)[:count]


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
        self.pending_commands: Dict[int, str] = {}
        
        # Main event loop reference
        self.main_loop = None
        
        # FEATURE: New Modules
        self.rate_limiter = RateLimiter()
        self.alias_manager = AliasManager("aliases.json")
        self.scheduler = AsyncIOScheduler()
        self.genai_stt_client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Build application
        self.application = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(self._post_init).build()
        self._register_handlers()
    
    async def _post_init(self, application: Application) -> None:
        self.main_loop = asyncio.get_running_loop()
        logger.info("Main event loop captured for thread-safe messaging")

    def _register_handlers(self):
        self.application.add_handler(CommandHandler("start", self.cmd_start))
        self.application.add_handler(CommandHandler("help", self.cmd_help))
        self.application.add_handler(CommandHandler("task", self.cmd_task))
        self.application.add_handler(CommandHandler("status", self.cmd_status))
        self.application.add_handler(CommandHandler("screenshot", self.cmd_screenshot))
        self.application.add_handler(CommandHandler("stop", self.cmd_stop))
        self.application.add_handler(CommandHandler("history", self.cmd_history))
        self.application.add_handler(CommandHandler("clear", self.cmd_clear))
        
        # FEATURE: New Commands
        self.application.add_handler(CommandHandler("alias", self.cmd_alias))
        self.application.add_handler(CommandHandler("sysinfo", self.cmd_sysinfo))
        self.application.add_handler(CommandHandler("clip", self.cmd_clip))
        self.application.add_handler(CommandHandler("webcam", self.cmd_webcam))
        self.application.add_handler(CommandHandler("getfile", self.cmd_getfile))
        self.application.add_handler(CommandHandler("schedule", self.cmd_schedule))
        
        # FEATURE: Voice & File Handlers
        self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
        self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        
        self.application.add_handler(CallbackQueryHandler(self.callback_handler))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text_message))
    
    # ============ AUTHORIZATION & RATE LIMITING ============
    
    def _is_authorized(self, user_id: int) -> bool:
        return user_id in AUTHORIZED_USERS
    
    async def _check_auth(self, update: Update) -> bool:
        if not self.main_loop:
            self.main_loop = asyncio.get_running_loop()
        
        user_id = update.effective_user.id
        
        # FEATURE: Rate Limiting Check
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ **Rate Limit Exceeded**\nPlease wait a minute before sending more commands.", parse_mode="Markdown")
            return False

        if not self._is_authorized(user_id):
            await update.message.reply_text("⛔ **UNAUTHORIZED**", parse_mode="Markdown")
            return False
        return True
    
    # ============ FEATURE: VOICE COMMANDS ============
    
    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        
        await update.message.reply_text("🎤 Processing voice message...")
        try:
            voice = update.message.voice or update.message.audio
            file = await voice.get_file()
            
            # Download to memory
            file_bytes = await file.download_as_bytearray()
            
            # Use Gemini for STT
            response = self.genai_stt_client.models.generate_content(
                model="gemini-2.0-flash", # 2.0 flash is highly optimized for audio
                contents=[
                    genai_types.Part.from_bytes(data=bytes(file_bytes), mime_type="audio/ogg"),
                    "Transcribe this audio accurately into plain text. Only output the spoken words, nothing else."
                ]
            )
            
            transcribed_text = response.text.strip()
            
            if transcribed_text:
                await update.message.reply_text(f"🗣️ *Heard:* \"{transcribed_text}\"\n\n⚙️ Executing...", parse_mode="Markdown")
                await self._process_command(transcribed_text, update)
            else:
                await update.message.reply_text("❌ Could not understand the audio.")
                
        except Exception as e:
            logger.error(f"Voice transcription error: {e}")
            await update.message.reply_text(f"❌ Voice processing failed: {e}")

    # ============ FEATURE: FILE TRANSFER ============
    
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        
        try:
            doc = update.message.document
            filename = doc.file_name
            
            # Sanitize filename to prevent path traversal
            safe_filename = os.path.basename(filename)
            save_path = os.path.join(DOWNLOADS_FOLDER, safe_filename)
            
            file = await doc.get_file()
            await file.download_to_drive(save_path)
            
            await update.message.reply_text(f"✅ File saved to Downloads:\n`{safe_filename}`", parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Failed to save file: {e}")

    async def cmd_getfile(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        if not context.args:
            await update.message.reply_text("Usage: `/getfile <filename>`\nGets file from your Downloads folder.", parse_mode="Markdown")
            return
        
        filename = " ".join(context.args)
        safe_filename = os.path.basename(filename)
        target_path = Path(DOWNLOADS_FOLDER) / safe_filename
        
        # Security: Resolve path to prevent directory traversal
        try:
            target_path.resolve().relative_to(Path(DOWNLOADS_FOLDER).resolve())
        except ValueError:
            await update.message.reply_text("⛔ Access denied: Invalid file path.")
            return
            
        if target_path.is_file():
            try:
                with open(target_path, "rb") as f:
                    await update.message.reply_document(f, caption=f"📂 {safe_filename}")
            except Exception as e:
                await update.message.reply_text(f"❌ Failed to send file: {e}")
        else:
            await update.message.reply_text(f"❌ File `{safe_filename}` not found in Downloads.")

    # ============ FEATURE: CUSTOM SHORTCUTS (ALIASES) ============
    
    async def cmd_alias(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        if not context.args:
            await update.message.reply_text("Usage:\n`/alias add <word> <full command>`\n`/alias del <word>`\n`/alias list`", parse_mode="Markdown")
            return
            
        action = context.args[0].lower()
        
        if action == "add" and len(context.args) >= 3:
            short = context.args[1]
            full_cmd = " ".join(context.args[2:])
            if self.alias_manager.add(short, full_cmd):
                await update.message.reply_text(f"✅ Alias saved: `{short}` → `{full_cmd}`", parse_mode="Markdown")
            else:
                await update.message.reply_text("❌ Alias must be a single word/character.")
                
        elif action == "del" and len(context.args) >= 2:
            short = context.args[1]
            if self.alias_manager.remove(short):
                await update.message.reply_text(f"🗑️ Alias `{short}` deleted.")
            else:
                await update.message.reply_text("❌ Alias not found.")
                
        elif action == "list":
            aliases = self.alias_manager.list_all()
            if not aliases:
                await update.message.reply_text("📭 No aliases saved.")
            else:
                msg = "📝 **Saved Aliases:**\n\n"
                for short, full in aliases.items():
                    msg += f"• `{short}` → {full[:50]}...\n" if len(full) > 50 else f"• `{short}` → {full}\n"
                await update.message.reply_text(msg, parse_mode="Markdown")
        else:
            await update.message.reply_text("❌ Invalid alias command.")

    # ============ FEATURE: SYSTEM MONITORING ============
    
    async def cmd_sysinfo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        
        def bar(percent, length=15):
            filled = int(percent / 100 * length)
            return "█" * filled + "░" * (length - filled)

        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('C:/')
        
        msg = (
            f"📊 **System Monitor**\n\n"
            f"🔥 CPU: {cpu}%\n{bar(cpu)}\n\n"
            f"💾 RAM: {ram.percent}% ({ram.used//1024//1024}MB / {ram.total//1024//1024}MB)\n{bar(ram.percent)}\n\n"
            f"💿 Disk C: {disk.percent}% ({disk.free//1024//1024//1024}GB Free)\n{bar(disk.percent)}"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    # ============ FEATURE: CLIPBOARD SYNC ============
    
    async def cmd_clip(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        
        if not context.args:
            # Read clipboard
            try:
                content = pyperclip.paste()
                if content:
                    await update.message.reply_text(f"📋 **Clipboard Content:**\n```\n{content[:1000]}\n```", parse_mode="Markdown")
                else:
                    await update.message.reply_text("📋 Clipboard is empty.")
            except Exception as e:
                await update.message.reply_text(f"❌ Failed to read clipboard: {e}")
        else:
            # Set clipboard
            text = " ".join(context.args)
            try:
                pyperclip.copy(text)
                await update.message.reply_text("✅ Clipboard updated!")
            except Exception as e:
                await update.message.reply_text(f"❌ Failed to set clipboard: {e}")

    # ============ FEATURE: WEBCAM CAPTURE ============
    
    async def cmd_webcam(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        
        await update.message.reply_text("📸 Accessing webcam...")
        try:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                path = "webcam_capture.jpg"
                cv2.imwrite(path, frame)
                with open(path, "rb") as photo:
                    await update.message.reply_photo(photo, caption="📸 Webcam Capture")
                os.remove(path)
            else:
                await update.message.reply_text("❌ Could not access webcam. Is it connected?")
        except Exception as e:
            await update.message.reply_text(f"❌ Webcam error: {e}")

    # ============ FEATURE: SCHEDULED TASKS ============
    
    async def cmd_schedule(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        
        if not context.args or context.args[0].lower() == "list":
            jobs = self.scheduler.get_jobs()
            if not jobs:
                await update.message.reply_text("📭 No scheduled tasks.")
            else:
                msg = "⏰ **Scheduled Tasks:**\n\n"
                for job in jobs:
                    msg += f"• `{job.args[0][:40]}` at {job.next_run_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                await update.message.reply_text(msg, parse_mode="Markdown")
                
        elif context.args[0].lower() == "clear":
            self.scheduler.remove_all_jobs()
            await update.message.reply_text("🗑️ All scheduled tasks cleared.")
            
        else:
            # Expected: /schedule HH:MM <command>
            try:
                time_str = context.args[0]
                cmd = " ".join(context.args[1:])
                
                # Parse time
                hour, minute = map(int, time_str.split(':'))
                now = datetime.now()
                run_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                if run_time < now:
                    run_time += timedelta(days=1) # Schedule for tomorrow if time passed today
                    
                chat_id = update.effective_chat.id
                self.scheduler.add_job(
                    self._execute_scheduled_task,
                    trigger='date',
                    run_date=run_time,
                    args=[cmd, chat_id]
                )
                await update.message.reply_text(f"⏰ Task scheduled for `{run_time.strftime('%H:%M')}`:\n`{cmd}`", parse_mode="Markdown")
            except Exception as e:
                await update.message.reply_text("❌ Invalid format. Use: `/schedule HH:MM <your task>`")

    async def _execute_scheduled_task(self, command: str, chat_id: int):
        """Triggered by APScheduler. Pushes directly to queue to avoid blocking and respect concurrency."""
        with self.lock:
            self.task_queue.put((command, chat_id))
        # Trigger queue processor safely
        if self.main_loop and not self.main_loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._async_process_queue(), self.main_loop)

    async def _async_process_queue(self):
        """Async wrapper to call _process_queue from scheduler thread"""
        self._process_queue()

    # ============ ORIGINAL COMMAND HANDLERS ============
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        await update.message.reply_text(
            "🤖 **PC Control Agent - Remote Interface**\n\n"
            "💬 Text/Voice → Execute task\n"
            "📂 Send File → Save to Downloads\n"
            "⏰ `/schedule HH:MM <task>`\n"
            "🔗 `/alias add <word> <cmd>`\n"
            "📊 `/sysinfo` | 📋 `/clip` | 📸 `/webcam`\n"
            "📂 `/getfile <name>` | 🖥️ `/screenshot`",
            parse_mode="Markdown"
        )
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        help_text = """
📖 **FULL HELP GUIDE**

**Core Tasks:**
• Text/Voice: Just say what you want
• `/task <text>` - Explicit task execution

**New Features:**
• `/sysinfo` - CPU/RAM/Disk monitor
• `/clip` - Read clipboard
• `/clip set <text>` - Write to clipboard
• `/webcam` - Take photo
• Send File - Saves to Downloads
• `/getfile <name>` - Retrieves from Downloads
• `/alias add n open notepad` - Create shortcut
• `/schedule 14:30 open chrome` - Schedule task

**Monitoring:**
• `/status` - Current task status
• `/screenshot` - See your screen
• `/history` - Past 10 tasks
"""
        await update.message.reply_text(help_text, parse_mode="Markdown")
    
    async def cmd_task(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        if not context.args:
            await update.message.reply_text("❌ Usage: `/task <your task here>`", parse_mode="Markdown")
            return
        command = " ".join(context.args)
        await self._process_command(command, update)
    
    async def handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        if update.effective_chat.id in self.pending_commands: return
        
        command = update.message.text.strip()
        
        # FEATURE: Alias Expansion (Check BEFORE length check)
        expanded_cmd = self.alias_manager.get(command)
        if expanded_cmd:
            command = expanded_cmd
        elif len(command) < 3:
            return # Ignore short non-alias messages
            
        await self._process_command(command, update)
    
    async def _process_command(self, command: str, update: Update):
        chat_id = update.effective_chat.id
        
        with self.lock:
            if self.is_busy:
                await update.message.reply_text(
                    f"⏳ **Busy**\nCurrently: `{self.current_task.command[:50]}...`\nQueued: `{command[:50]}...`",
                    parse_mode="Markdown"
                )
                self.task_queue.put((command, chat_id))
                return
        
        if REQUIRE_CONFIRMATION:
            keyboard = [[InlineKeyboardButton("✅ Execute", callback_data="confirm_yes"), InlineKeyboardButton("❌ Cancel", callback_data="confirm_no")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            self.pending_commands[chat_id] = command
            display_cmd = command[:100] + "..." if len(command) > 100 else command
            await update.message.reply_text(f"📋 **Task:**\n`{display_cmd}`\n\nProceed?", parse_mode="Markdown", reply_markup=reply_markup)
        else:
            await self._start_task(command, chat_id)
    
    async def callback_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        chat_id = query.message.chat_id
        
        if query.data == "confirm_yes":
            command = self.pending_commands.pop(chat_id, None)
            if command:
                await query.edit_message_text(f"✅ **Confirmed!**\nStarting: `{command[:50]}...`", parse_mode="Markdown")
                await self._start_task(command, chat_id)
            else:
                await query.edit_message_text("❌ Command expired.")
        elif query.data == "confirm_no":
            self.pending_commands.pop(chat_id, None)
            await query.edit_message_text("❌ Task cancelled.")
    
    async def _start_task(self, command: str, chat_id: int):
        with self.lock:
            self.is_busy = True
            self.stop_flag.clear()
            task = self.task_history.create_task(command)
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()
            self.task_history.update_task(task)
            self.current_task = task
        
        self.worker_thread = threading.Thread(target=self._execute_task_thread, args=(command, chat_id, task.task_id), daemon=True)
        self.worker_thread.start()
    
    def _handle_task_timeout(self, chat_id: int, task_id: int):
        with self.lock:
            if self.is_busy and self.current_task and self.current_task.task_id == task_id:
                self.stop_flag.set()
                logger.warning(f"Task #{task_id} timed out.")
                self._send_safe(self._send_message(chat_id, f"⏰ **TASK TIMEOUT**\nTask #{task_id} exceeded limit.", parse_mode="Markdown"))

    def _execute_task_thread(self, command: str, chat_id: int, task_id: int):
        timeout_timer = threading.Timer(TASK_TIMEOUT_SECONDS, self._handle_task_timeout, args=[chat_id, task_id])
        timeout_timer.daemon = True
        timeout_timer.start()
        
        try:
            results = self.pc_agent.execute_task(command)
            
            if self.stop_flag.is_set():
                self.current_task.status = TaskStatus.STOPPED
                self.current_task.completed_at = datetime.now().isoformat()
                self.task_history.update_task(self.current_task)
                self._send_safe(self._send_message(chat_id, f"🛑 **TASK STOPPED**\nTask #{task_id} stopped.", parse_mode="Markdown"))
            else:
                self.current_task.status = TaskStatus.COMPLETED if results["success"] else TaskStatus.FAILED
                self.current_task.completed_at = datetime.now().isoformat()
                self.current_task.steps_completed = results["steps_completed"]
                self.current_task.total_steps = results["total_steps"]
                self.current_task.replans = results.get("replans", 0)
                self.current_task.result_summary = self._summarize_results(results)
                self.task_history.update_task(self.current_task)
                self._send_safe(self._send_completion_message(chat_id, task_id, results))
        
        except Exception as e:
            self.current_task.status = TaskStatus.FAILED
            self.current_task.completed_at = datetime.now().isoformat()
            self.current_task.error = str(e)
            self.task_history.update_task(self.current_task)
            self._send_safe(self._send_message(chat_id, f"❌ **TASK ERROR**\n`{str(e)[:500]}`", parse_mode="Markdown"))
        
        finally:
            timeout_timer.cancel()
            with self.lock:
                self.is_busy = False
                self.current_task = None
            self._process_queue()
    
    def _summarize_results(self, results: Dict) -> str:
        summary_parts = []
        if results.get("step_results"):
            for sr in results["step_results"]:
                status = "✓" if sr["success"] else "✗"
                summary_parts.append(f"{status} {sr['description']}")
        return "\n".join(summary_parts[:20])
    
    async def _send_completion_message(self, chat_id: int, task_id: int, results: Dict):
        status_emoji = "✅" if results["success"] else "❌"
        status_text = "COMPLETED" if results["success"] else "FAILED/ABORTED"
        message = (f"{status_emoji} **TASK {status_text}**\n\n📋 Task #{task_id}\n📊 Steps: {results['steps_completed']}/{results['total_steps']}\n🔄 Re-plans: {results.get('replans', 0)}\n")
        
        if results.get("step_results"):
            message += "\n**Steps:**\n"
            for sr in results["step_results"][-10:]:
                status = "✓" if sr["success"] else "✗"
                message += f"  {status} {sr['description']}\n"
        
        await self._send_message(chat_id, message, parse_mode="Markdown")
        
        try:
            screenshot = self.pc_agent.take_screenshot("final_result.png")
            with open("final_result.png", "rb") as photo:
                await self._send_photo(chat_id, photo, caption="📸 Final screen state:")
            self.pc_agent.cleanup_screenshot("final_result.png")
        except Exception as e:
            pass
    
    async def _send_message(self, chat_id: int, text: str, parse_mode: Optional[str] = None):
        try:
            await self.application.bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def _send_photo(self, chat_id: int, photo: Any, caption: Optional[str] = None):
        try:
            await self.application.bot.send_photo(chat_id=chat_id, photo=photo, caption=caption)
        except Exception as e:
            logger.error(f"Error sending photo: {e}")

    def _send_safe(self, coro):
        if self.main_loop and not self.main_loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(coro, self.main_loop)
            future.add_done_callback(lambda fut: fut.result() if not fut.exception() else logger.error(f"Failed: {fut.exception()}"))
        else:
            logger.error("Event loop not available, message dropped!")
    
    def _process_queue(self):
        while not self.task_queue.empty():
            try:
                command, chat_id = self.task_queue.get_nowait()
                with self.lock:
                    if self.is_busy:
                        self.task_queue.put((command, chat_id))
                        break
                self._send_safe(self._start_task(command, chat_id))
                break
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Failed to process queued task: {e}")
    
    # ============ MONITORING COMMANDS ============
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        with self.lock:
            if self.is_busy and self.current_task:
                elapsed = ""
                if self.current_task.started_at:
                    start = datetime.fromisoformat(self.current_task.started_at)
                    elapsed = str(datetime.now() - start).split('.')[0]
                status_msg = (f"⏳ **WORKING**\n\n📋 Task #{self.current_task.task_id}\n📝 `{self.current_task.command[:100]}`\n⏱️ Elapsed: {elapsed}\n📥 Queue: {self.task_queue.qsize()}")
            else:
                status_msg = f"😴 **IDLE**\n📥 Queue: {self.task_queue.qsize()}"
        await update.message.reply_text(status_msg, parse_mode="Markdown")
    
    async def cmd_screenshot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        if not ALLOW_SCREENSHOT_COMMAND: return
        await update.message.reply_text("📸 Taking screenshot...")
        try:
            path = "remote_screenshot.png"
            self.pc_agent.take_screenshot(path)
            with open(path, "rb") as photo:
                await update.message.reply_photo(photo=photo, caption=f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.pc_agent.cleanup_screenshot(path)
        except Exception as e:
            await update.message.reply_text(f"❌ Failed: {e}")
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        if not EMERGENCY_STOP_ENABLED: return
        with self.lock:
            if self.is_busy:
                self.stop_flag.set()
                import pyautogui
                pyautogui.moveTo(0, 0)
                await update.message.reply_text("🛑 **EMERGENCY STOP ACTIVATED**", parse_mode="Markdown")
            else:
                await update.message.reply_text("😊 Nothing is running.")
        while not self.task_queue.empty():
            try: self.task_queue.get_nowait()
            except queue.Empty: break
    
    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        recent = self.task_history.get_recent(10)
        if not recent: return await update.message.reply_text("📭 No task history yet.")
        message = "📋 **Recent Tasks:**\n\n"
        for task in recent:
            status_emoji = {"completed": "✅", "failed": "❌", "stopped": "🛑", "running": "⏳", "queued": "📥"}.get(task.status.value, "❓")
            cmd_short = task.command[:40] + "..." if len(task.command) > 40 else task.command
            message += f"{status_emoji} `#{task.task_id}` {cmd_short}\n"
        await update.message.reply_text(message, parse_mode="Markdown")
    
    async def cmd_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        self.task_history.tasks = {}
        self.task_history._next_id = 1
        self.task_history._save()
        await update.message.reply_text("🗑️ Task history cleared.")

    # ============ FEATURE: SELF-RECONNECTION ============
    
    def _check_internet(self) -> bool:
        try:
            requests.get("https://api.telegram.org", timeout=5)
            return True
        except (requests.ConnectionError, requests.Timeout):
            return False

    async def _bootstrap(self):
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
        if not self.scheduler.running:
            self.scheduler.start()

    async def _shutdown_app(self):
        try:
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)
            if self.application.updater.running:
                await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def run(self):
        print("=" * 60)
        print("PC CONTROL AGENT - ADVANCED TELEGRAM INTERFACE")
        print("=" * 60)
        print(f"🛡️ Autonomous Reconnect: ENABLED")
        print(f"🚦 Rate Limiting: ENABLED")
        print(f"⏰ Task Scheduler: ENABLED")
        print("=" * 60)
        
        retry_delay = 10

        while True:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                print("🌐 Connecting to Telegram...")
                loop.run_until_complete(self._bootstrap())
                print("✅ Bot is running and polling.")
                retry_delay = 10 # Reset on successful connection
                loop.run_forever()
                
            except KeyboardInterrupt:
                print("\n⛔ Ctrl+C detected. Shutting down permanently...")
                loop.run_until_complete(self._shutdown_app())
                break
                
            except Exception as e:
                print(f"\n❌ Connection Lost: {e}")
                print(f"💤 Cleaning up and sleeping for {retry_delay} seconds...")
                
            finally:
                try:
                    loop.run_until_complete(self._shutdown_app())
                except Exception:
                    pass
                loop.close()

            # Sleep & Recovery Phase
            print("🔍 Checking internet connection...")
            while not self._check_internet():
                print(f"⏳ No internet. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                if retry_delay < 60:
                    retry_delay += 10 # Exponential backoff up to 60s
            
            print("🌐 Internet detected! Rebooting bot...")
            time.sleep(2) # Brief pause to let connection stabilize