"""
Telegram Bot Interface for Remote PC Control
Handles all communication between user and agent
Enhanced with: Self-Reconnect, Rate Limiting, Voice Commands, Scheduled Tasks,
               Aliases, File Transfer, System Monitoring, Clipboard Sync, Webcam Capture,
               Notification Forwarding, Screen Streaming, PC Popup Dialogs, Mic Recording.
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
import pyautogui
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
from Core.core_agent import PCControlAgent
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

# Fallbacks for NEW feature configs (prevents errors if config.py isn't updated yet)
try:
    ENABLE_NOTIFICATION_FORWARDING
except NameError:
    ENABLE_NOTIFICATION_FORWARDING = False

try:
    SCREEN_STREAM_DURATION
except NameError:
    SCREEN_STREAM_DURATION = 10

try:
    SCREEN_STREAM_FPS
except NameError:
    SCREEN_STREAM_FPS = 10.0

try:
    MIC_RECORD_DURATION
except NameError:
    MIC_RECORD_DURATION = 10

try:
    MIC_SAMPLE_RATE
except NameError:
    MIC_SAMPLE_RATE = 44100

# Check for Windows specific GUI hooking
try:
    import win32gui
    HAS_WIN32GUI = True
except ImportError:
    HAS_WIN32GUI = False

# Set up logging (Fix #15)
logger = logging.getLogger(__name__)


# ==========================================
# FEATURE: Notification Forwarder (Win32)
# ==========================================
class NotificationListener:
    """Polls for Windows Toast Notifications and forwards them."""
    def __init__(self, callback):
        self.callback = callback
        self.seen_windows = set()
        self.running = True
        
        if HAS_WIN32GUI:
            self.thread = threading.Thread(target=self._poll, daemon=True)
            self.thread.start()
            logger.info("🔔 Notification Forwarding Listener Started.")
        else:
            logger.warning("win32gui missing, notification forwarding disabled.")

    def _poll(self):
        while self.running:
            def enum_callback(hwnd, _):
                try:
                    if not win32gui.IsWindowVisible(hwnd):
                        return
                    
                    class_name = win32gui.GetClassName(hwnd)
                    # Windows 11 Toast/Notification Classes
                    if class_name in ["Windows.UI.Core.CoreWindow", "Xaml_WindowedPopupClass", "ToastDialog"]:
                        title = win32gui.GetWindowText(hwnd)
                        if title and title.strip() and hwnd not in self.seen_windows:
                            self.seen_windows.add(hwnd)
                            self.callback(title.strip())
                except Exception:
                    pass
            
            try:
                win32gui.EnumWindows(enum_callback, None)
                # Clean up handles of windows that have closed
                self.seen_windows = {hwnd for hwnd in self.seen_windows if win32gui.IsWindow(hwnd)}
            except Exception:
                pass
            
            time.sleep(1.5) # Poll every 1.5 seconds to catch 5-second toasts reliably


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
        if len(short.split()) > 1: return False
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
        
        # FEATURE: Notification Forwarding Init
        self.notif_listener = None
        if ENABLE_NOTIFICATION_FORWARDING and HAS_WIN32GUI:
            self.notif_listener = NotificationListener(self._forward_notification)
        
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
        
        # FEATURE: Existing Commands
        self.application.add_handler(CommandHandler("alias", self.cmd_alias))
        self.application.add_handler(CommandHandler("sysinfo", self.cmd_sysinfo))
        self.application.add_handler(CommandHandler("clip", self.cmd_clip))
        self.application.add_handler(CommandHandler("webcam", self.cmd_webcam))
        self.application.add_handler(CommandHandler("getfile", self.cmd_getfile))
        self.application.add_handler(CommandHandler("schedule", self.cmd_schedule))
        
        # FEATURE: New Commands (Non-conflicting)
        self.application.add_handler(CommandHandler("stream", self.cmd_stream))
        self.application.add_handler(CommandHandler("popup", self.cmd_popup))
        self.application.add_handler(CommandHandler("mic", self.cmd_mic))
        
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
        
        if not self.rate_limiter.is_allowed(user_id):
            await update.message.reply_text("⚠️ **Rate Limit Exceeded**\nPlease wait a minute before sending more commands.", parse_mode="Markdown")
            return False

        if not self._is_authorized(user_id):
            await update.message.reply_text("⛔ **UNAUTHORIZED**", parse_mode="Markdown")
            return False
        return True

    # ============ FEATURE: NOTIFICATION FORWARDING (PC -> PHONE) ============
    
    def _forward_notification(self, text: str):
        """Callback triggered by NotificationListener in a background thread."""
        text = text[:500] # Prevent massive messages
        msg = f"🔔 **PC Notification:**\n```\n{text}\n```"
        self._send_safe(self._broadcast_to_admins(msg, parse_mode="Markdown"))
        
    async def _broadcast_to_admins(self, text: str, parse_mode: Optional[str] = None):
        for uid in AUTHORIZED_USERS:
            try:
                await self.application.bot.send_message(chat_id=uid, text=text, parse_mode=parse_mode)
            except Exception as e:
                logger.error(f"Failed to broadcast notif to {uid}: {e}")

    # ============ FEATURE: LIVE SCREEN STREAMING ============
    
    async def cmd_stream(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        
        await update.message.reply_text(f"🎥 Recording screen for {SCREEN_STREAM_DURATION} seconds...")
        
        try:
            import numpy as np
            from PIL import ImageGrab
            
            width, height = pyautogui.size()
            filename = "screen_stream.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, SCREEN_STREAM_FPS, (width, height))
            
            frames = int(SCREEN_STREAM_DURATION * SCREEN_STREAM_FPS)
            for _ in range(frames):
                # ImageGrab is much faster than pyautogui.screenshot for video
                img = ImageGrab.grab(bbox=(0, 0, width, height))
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                out.write(frame)
                await asyncio.sleep(1.0 / SCREEN_STREAM_FPS) # Non-blocking sleep
                
            out.release()
            
            with open(filename, "rb") as video:
                await update.message.reply_video(video, caption=f"🎥 Screen Stream ({SCREEN_STREAM_DURATION}s)")
                
            if os.path.exists(filename):
                os.remove(filename)
                
        except Exception as e:
            logger.error(f"Screen recording failed: {e}")
            await update.message.reply_text(f"❌ Screen recording failed: {e}")

    # ============ FEATURE: MESSAGING DIALOG BOX ============
    
    def _show_popup_thread(self, message: str):
        """Spawns a native unclosable tkinter popup. Must run in its own thread."""
        try:
            import tkinter as tk
            
            root = tk.Tk()
            root.attributes("-topmost", True)
            root.overrideredirect(True) # Removes the X button and borders completely
            
            # Modern dark theme styling
            frame = tk.Frame(root, bg="#2b2b2b", bd=0)
            frame.pack(padx=0, pady=0, fill="both", expand=True)
            
            tk.Label(frame, text="📱 Telegram Message", font=("Segoe UI", 10, "bold"), bg="#2b2b2b", fg="#ffffff").pack(padx=20, pady=(15, 5), anchor="w")
            tk.Label(frame, text=message, font=("Segoe UI", 11), bg="#2b2b2b", fg="#e0e0e0", wraplength=350, justify="left").pack(padx=20, pady=5, anchor="w")
            
            def on_ok():
                root.destroy()
                
            btn_frame = tk.Frame(frame, bg="#2b2b2b")
            btn_frame.pack(pady=(10, 15), padx=20, anchor="e")
            tk.Button(btn_frame, text="OK", command=on_ok, width=8, bg="#0078d4", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2").pack()
            
            # Center on screen
            root.update_idletasks()
            w = root.winfo_width()
            h = root.winfo_height()
            x = (root.winfo_screenwidth() // 2) - (w // 2)
            y = (root.winfo_screenheight() // 2) - (h // 2)
            root.geometry(f'+{x}+{y}')
            
            root.mainloop()
        except Exception as e:
            logger.error(f"Failed to show popup (No active GUI session?): {e}")

    async def cmd_popup(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        
        if not context.args:
            await update.message.reply_text("Usage: `/popup <your message>`", parse_mode="Markdown")
            return
            
        message = " ".join(context.args)
        
        threading.Thread(target=self._show_popup_thread, args=(message,), daemon=True).start()
        await update.message.reply_text("💻 Popup shown on PC screen.")

    # ============ FEATURE: MICROPHONE RECORDING ============
    
    async def cmd_mic(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        
        await update.message.reply_text(f"🎙️ Recording microphone for {MIC_RECORD_DURATION} seconds...")
        
        try:
            import sounddevice as sd
            import soundfile as sf
            
            filename = "mic_recording.wav"
            fs = MIC_SAMPLE_RATE
            seconds = MIC_RECORD_DURATION
            
            mydata = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
            
            # Run waiting in executor so it doesn't block the asyncio event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, sd.wait)
            
            sf.write(filename, mydata, fs)
            
            with open(filename, "rb") as audio:
                await update.message.reply_audio(audio, caption=f"🎙️ Mic Recording ({seconds}s)")
                
            if os.path.exists(filename):
                os.remove(filename)
                
        except ImportError:
            await update.message.reply_text("❌ Missing dependencies for Mic.\nPlease run: `pip install sounddevice soundfile`")
        except Exception as e:
            logger.error(f"Mic recording failed: {e}")
            await update.message.reply_text(f"❌ Mic recording failed: {e}")

    # ============ FEATURE: VOICE COMMANDS ============
    
    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update):
            return
        
        await update.message.reply_text("🎤 Processing voice message...")
        try:
            voice = update.message.voice or update.message.audio
            file = await voice.get_file()
            file_bytes = await file.download_as_bytearray()
            
            response = self.genai_stt_client.models.generate_content(
                model="gemini-2.0-flash", 
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
        if not await self._check_auth(update): return
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
        if not await self._check_auth(update): return
        
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
        if not await self._check_auth(update): return
        
        if not context.args:
            try:
                content = pyperclip.paste()
                if content:
                    await update.message.reply_text(f"📋 **Clipboard Content:**\n```\n{content[:1000]}\n```", parse_mode="Markdown")
                else:
                    await update.message.reply_text("📋 Clipboard is empty.")
            except Exception as e:
                await update.message.reply_text(f"❌ Failed to read clipboard: {e}")
        else:
            text = " ".join(context.args)
            try:
                pyperclip.copy(text)
                await update.message.reply_text("✅ Clipboard updated!")
            except Exception as e:
                await update.message.reply_text(f"❌ Failed to set clipboard: {e}")

    # ============ FEATURE: WEBCAM CAPTURE ============
    
    async def cmd_webcam(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        
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
        if not await self._check_auth(update): return
        
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
            try:
                time_str = context.args[0]
                cmd = " ".join(context.args[1:])
                
                hour, minute = map(int, time_str.split(':'))
                now = datetime.now()
                run_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                if run_time < now:
                    run_time += timedelta(days=1) 
                    
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
        with self.lock:
            self.task_queue.put((command, chat_id))
        if self.main_loop and not self.main_loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._async_process_queue(), self.main_loop)

    async def _async_process_queue(self):
        self._process_queue()

    # ============ ORIGINAL COMMAND HANDLERS ============
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        
        notif_status = "ACTIVE 🔔" if self.notif_listener else "DISABLED"
        await update.message.reply_text(
            "🤖 **PC Control Agent - Remote Interface**\n\n"
            "💬 Text/Voice → Execute task\n"
            "📂 Send File → Save to Downloads\n"
            "⏰ `/schedule HH:MM <task>`\n"
            "🔗 `/alias add <word> <cmd>`\n"
            "📊 `/sysinfo` | 📋 `/clip` | 📸 `/webcam`\n"
            "📂 `/getfile <name>` | 🖥️ `/screenshot`\n"
            "🎥 `/stream` | 🎙️ `/mic` | 💬 `/popup <msg>`\n\n"
            f"🔔 Notification Forwarding: {notif_status}",
            parse_mode="Markdown"
        )
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self._check_auth(update): return
        help_text = """
📖 **FULL HELP GUIDE**

**Core Tasks:**
• Text/Voice: Just say what you want
• `/task <text>` - Explicit task execution

**Utilities:**
• `/sysinfo` - CPU/RAM/Disk monitor
• `/clip` - Read clipboard
• `/clip set <text>` - Write to clipboard
• `/webcam` - Take photo
• Send File - Saves to Downloads
• `/getfile <name>` - Retrieves from Downloads
• `/alias add n open notepad` - Create shortcut
• `/schedule 14:30 open chrome` - Schedule task

**NEW Features:**
• `/stream` - 10s screen recording video
• `/mic` - 10s microphone audio recording
• `/popup <message>` - Show unclosable popup on PC

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
        
        expanded_cmd = self.alias_manager.get(command)
        if expanded_cmd:
            command = expanded_cmd
        elif len(command) < 3:
            return 
            
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
        except Exception:
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
            if self.notif_listener:
                self.notif_listener.running = False
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
        print(f"🔔 Notification Forwarding: {'ACTIVE' if self.notif_listener else 'DISABLED'}")
        print("=" * 60)
        
        retry_delay = 10

        while True:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                print("🌐 Connecting to Telegram...")
                loop.run_until_complete(self._bootstrap())
                print("✅ Bot is running and polling.")
                retry_delay = 10 
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

            print("🔍 Checking internet connection...")
            while not self._check_internet():
                print(f"⏳ No internet. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                if retry_delay < 60:
                    retry_delay += 10 
            
            print("🌐 Internet detected! Rebooting bot...")
            time.sleep(2) 