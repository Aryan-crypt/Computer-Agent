"""
PC Control AI Agent with Multi-Model Architecture and UI
Uses Gemini 2.5 Pro for planning, Holo 1.5 for coordinate extraction, 
and Pollinations OpenAI for content reading/writing
"""

import os
import sys
import time
import json
import base64
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import threading
import queue
from DataBase.shortcuts import WINDOWS_SHORTCUTS

# Core libraries
import pyautogui
from PIL import Image, ImageDraw
import pytesseract
import keyboard
import mouse
import win32gui
import win32con
from io import BytesIO

# AI Model libraries
import google.generativeai as genai
from gradio_client import Client, handle_file
import requests
import urllib.parse
from API import GEMINI_API_KEY

# PyQt5 for UI
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QPushButton, QLabel, 
                             QScrollArea, QFrame, QLineEdit, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect
from PyQt5.QtGui import QFont, QTextCursor, QColor, QPalette

# Configure pyautogui for safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

# Configuration
POLLINATIONS_BASE_URL = "https://text.pollinations.ai/openai"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

@dataclass
class Step:
    """Represents a single step in the task execution"""
    description: str
    action_type: str
    parameters: Dict[str, Any]
    completed: bool = False
    result: Optional[str] = None

class ActionType(Enum):
    """Types of actions the agent can perform"""
    KEYBOARD = "keyboard"
    MOUSE_CLICK = "mouse_click"
    MOUSE_MOVE = "mouse_move"
    MOUSE_SCROLL = "mouse_scroll"
    TYPE_TEXT = "type_text"
    READ_CONTENT = "read_content"
    WAIT = "wait"
    SCREENSHOT = "screenshot"

class CommanderAgent:
    """Gemini-based planning and command agent"""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        self.shortcuts_knowledge = json.dumps(WINDOWS_SHORTCUTS, indent=2)
        
    def plan_task(self, user_prompt: str, current_screenshot: Optional[Image.Image] = None) -> List[Step]:
        """Generate a plan for completing the user's task"""
        
        system_prompt = f"""You are an Very Clever and Smart AI assistant that controls a Windows 11 PC. 
        You need to plan steps to complete the user's task.
        
        Available Windows 11 and Google Chrome shortcuts:
        {self.shortcuts_knowledge}
        
        When planning:
        1. Break down the task into small, atomic steps
        2. Prefer keyboard shortcuts when possible
        3. Always before typing on an input area, click on that input area before typing
        4. For each step, specify if it needs:
           - Keyboard shortcut
           - Mouse click (will need coordinate extraction)
           - Text input
           - Content reading from screen
        5. Tips :
            - Claculator can be used through Keyboard (e.g; using numbers (1,2,3,4,5,6,7,8,9,0), division (/), multiplication (*), addition (+), subtraction (-) and results or equals to (=)).
            - For enabling the searchbar in youtube click '/' then type the search query.
        6. Remember :
            - Do not copy or select any text, because you are not allowed to select text rather than read_content then type_text.

        Return a JSON array of steps with format:
        [
            {{
                "description": "Step description",
                "action_type": "keyboard|mouse_click|type_text|read_content|wait",
                "parameters": {{
                    "keys": ["key1", "key2"],  // for keyboard
                    "target": "description of click target in detailed way",  // for mouse_click
                    "text": "text to type",  // for type_text
                    "duration": 1.0  // for wait
                }}
            }}
        ]
        
        User task: {user_prompt}
        """
        
        try:
            if current_screenshot:
                img_byte_arr = BytesIO()
                current_screenshot.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                response = self.model.generate_content([
                    system_prompt,
                    {"mime_type": "image/png", "data": img_byte_arr}
                ])
            else:
                response = self.model.generate_content(system_prompt)
            
            response_text = response.text
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                steps_data = json.loads(json_match.group())
                return [Step(**step) for step in steps_data]
            else:
                print(f"Could not parse plan from response: {response_text}")
                return []
                
        except Exception as e:
            print(f"Error in planning: {e}")
            return []
    
    def analyze_screenshot(self, screenshot: Image.Image, target_description: str) -> Dict[str, Any]:
        """Analyze screenshot to determine next action"""
        
        prompt = f"""Analyze this screenshot and help me find: {target_description}
        
        Determine:
        1. Is the target visible on screen (Taskbar or desktop)?
        2. Can this be achieved with a keyboard shortcut?
        3. If mouse click is needed, describe exactly where to click
        
        Return JSON:
        {{
            "target_found": true/false,
            "use_keyboard": true/false,
            "keyboard_shortcut": ["key1", "key2"] or null,
            "click_description": "detailed description of where to click" or null
        }}
        """
        
        try:
            img_byte_arr = BytesIO()
            screenshot.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            response = self.model.generate_content([
                prompt,
                {"mime_type": "image/png", "data": img_byte_arr}
            ])
            
            response_text = response.text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"target_found": False}
                
        except Exception as e:
            print(f"Error analyzing screenshot: {e}")
            return {"target_found": False}

class CoordinateExtractor:
    """Holo 1.5 model for extracting coordinates from screenshots"""
    
    def __init__(self):
        try:
            self.client = Client("Hcompany/Holo1.5-Localization")
        except Exception as e:
            print(f"Warning: Could not initialize Holo 1.5 client: {e}")
            self.client = None
    
    def extract_coordinates(self, screenshot_path: str, target_description: str) -> Optional[Tuple[int, int]]:
        """Extract coordinates of target element from screenshot"""
        
        if not self.client:
            print("Holo 1.5 not available, using fallback coordinate extraction")
            return self._fallback_extraction(screenshot_path, target_description)
        
        try:
            result = self.client.predict(
                input_numpy_image=handle_file(screenshot_path),
                task=f"Find and click on: {target_description}",
                api_name="/localize"
            )
            
            coord_pattern = r'(?:x[:\s]*)?(\d+)[,\s]+(?:y[:\s]*)?(\d+)'
            match = re.search(coord_pattern, str(result), re.IGNORECASE)
            
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                return (x, y)
            else:
                print(f"Could not parse coordinates from: {result}")
                return None
                
        except Exception as e:
            print(f"Error extracting coordinates: {e}")
            return self._fallback_extraction(screenshot_path, target_description)
    
    def _fallback_extraction(self, screenshot_path: str, target_description: str) -> Optional[Tuple[int, int]]:
        """Fallback method using OCR and pattern matching"""
        try:
            img = Image.open(screenshot_path)
            return (img.width // 2, img.height // 2)
        except:
            return None

class ContentProcessor:
    """Pollinations OpenAI model for reading and writing content"""
    
    def __init__(self):
        self.base_url = POLLINATIONS_BASE_URL
        
    def read_content(self, screenshot: Image.Image, instruction: str = "Read all text from this image") -> str:
        """Read and extract content from screenshot"""
        
        buffered = BytesIO()
        screenshot.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": "openai",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error reading content: {e}")
            return self._ocr_fallback(screenshot)
    
    def _ocr_fallback(self, screenshot: Image.Image) -> str:
        """Fallback OCR method"""
        try:
            text = pytesseract.image_to_string(screenshot)
            return text.strip()
        except Exception as e:
            print(f"OCR fallback failed: {e}")
            return ""

class PCControlAgent:
    """Main orchestrator agent that coordinates all components"""
    
    def __init__(self, status_callback=None):
        self.commander = CommanderAgent()
        self.coordinate_extractor = CoordinateExtractor()
        self.content_processor = ContentProcessor()
        self.current_screenshot = None
        self.screenshot_history = []
        self.temp_screenshots = []
        self.status_callback = status_callback
        
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.5
    
    def emit_status(self, message: str, status_type: str = "info"):
        """Emit status message to UI"""
        if self.status_callback:
            self.status_callback(message, status_type)
        
    def take_screenshot(self, save_path: Optional[str] = None) -> Image.Image:
        """Take a screenshot of the entire screen"""
        screenshot = pyautogui.screenshot()
        self.current_screenshot = screenshot
        self.screenshot_history.append(screenshot)
        
        if save_path:
            screenshot.save(save_path)
            self.temp_screenshots.append(save_path)
            
        if len(self.screenshot_history) > 10:
            self.screenshot_history.pop(0)
            
        return screenshot
    
    def cleanup_screenshot(self, filepath: str):
        """Delete a single screenshot file"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                if filepath in self.temp_screenshots:
                    self.temp_screenshots.remove(filepath)
        except Exception as e:
            print(f"Warning: Could not delete {filepath}: {e}")
    
    def cleanup_all_screenshots(self):
        """Delete all temporary screenshots"""
        for filepath in self.temp_screenshots[:]:
            self.cleanup_screenshot(filepath)
        self.temp_screenshots.clear()
    
    def execute_keyboard_action(self, keys: List[str]) -> bool:
        """Execute a keyboard shortcut or key combination"""
        try:
            if len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"Error executing keyboard action {keys}: {e}")
            return False
    
    def execute_mouse_click(self, x: int, y: int, button: str = 'left') -> bool:
        """Execute a mouse click at specified coordinates"""
        try:
            pyautogui.click(x, y, button=button)
            time.sleep(0.3)
            return True
        except Exception as e:
            print(f"Error executing mouse click at ({x}, {y}): {e}")
            return False
    
    def execute_mouse_scroll(self, amount: int) -> bool:
        """Execute mouse scroll"""
        try:
            pyautogui.scroll(amount)
            time.sleep(0.3)
            return True
        except Exception as e:
            print(f"Error executing scroll: {e}")
            return False
    
    def type_text(self, text: str, interval: float = 0.05) -> bool:
        """Type text with specified interval between keystrokes"""
        try:
            pyautogui.typewrite(text, interval=interval)
            time.sleep(0.3)
            return True
        except Exception as e:
            print(f"Error typing text: {e}")
            return False
    
    def execute_step(self, step: Step) -> bool:
        """Execute a single step in the plan"""
        
        self.emit_status(f"Executing: {step.description}", "executing")
        
        temp_screenshot_path = None
        
        try:
            if step.action_type == "keyboard":
                keys = step.parameters.get("keys", [])
                success = self.execute_keyboard_action(keys)
                
            elif step.action_type == "mouse_click":
                temp_screenshot_path = "temp_screenshot.png"
                screenshot = self.take_screenshot(temp_screenshot_path)
                target = step.parameters.get("target", "")
                
                coords = self.coordinate_extractor.extract_coordinates(temp_screenshot_path, target)
                
                if coords:
                    success = self.execute_mouse_click(coords[0], coords[1])
                else:
                    analysis = self.commander.analyze_screenshot(screenshot, target)
                    if analysis.get("use_keyboard") and analysis.get("keyboard_shortcut"):
                        success = self.execute_keyboard_action(analysis["keyboard_shortcut"])
                    else:
                        success = False
                
                if temp_screenshot_path:
                    self.cleanup_screenshot(temp_screenshot_path)
                        
            elif step.action_type == "type_text":
                text = step.parameters.get("text", "")
                success = self.type_text(text)
                
            elif step.action_type == "read_content":
                screenshot = self.take_screenshot()
                instruction = step.parameters.get("instruction", "Read all text from screen")
                content = self.content_processor.read_content(screenshot, instruction)
                step.result = content
                success = True
                
            elif step.action_type == "wait":
                duration = step.parameters.get("duration", 1.0)
                time.sleep(duration)
                success = True
                
            else:
                success = False
                
            step.completed = success
            
            if success:
                self.emit_status(f"✓ Completed: {step.description}", "success")
            else:
                self.emit_status(f"✗ Failed: {step.description}", "error")
            
            return success
            
        except Exception as e:
            self.emit_status(f"✗ Error: {step.description} - {str(e)}", "error")
            if temp_screenshot_path:
                self.cleanup_screenshot(temp_screenshot_path)
            step.completed = False
            return False
    
    def execute_task(self, user_prompt: str) -> Dict[str, Any]:
        """Main method to execute a complete task"""
        
        self.emit_status(f"Starting task: {user_prompt}", "info")
        
        # Minimize all windows except our UI
        self.emit_status("Minimizing windows...", "info")
        self.execute_keyboard_action(["win", "m"])
        time.sleep(1)
        
        initial_screenshot = self.take_screenshot("initial_state.png")
        
        self.emit_status("Generating execution plan...", "info")
        steps = self.commander.plan_task(user_prompt, initial_screenshot)
        
        self.cleanup_screenshot("initial_state.png")
        
        if not steps:
            self.emit_status("Failed to generate plan", "error")
            return {"success": False, "error": "Could not generate execution plan"}
        
        self.emit_status(f"Plan generated: {len(steps)} steps", "success")
        
        results = {
            "success": True,
            "steps_completed": 0,
            "total_steps": len(steps),
            "step_results": []
        }
        
        for i, step in enumerate(steps, 1):
            self.emit_status(f"[Step {i}/{len(steps)}]", "info")
            
            step_screenshot_path = f"step_{i}_before.png"
            self.take_screenshot(step_screenshot_path)
            
            success = self.execute_step(step)
            
            self.cleanup_screenshot(step_screenshot_path)
            
            results["step_results"].append({
                "step": i,
                "description": step.description,
                "success": success,
                "result": step.result
            })
            
            if success:
                results["steps_completed"] += 1
            
            time.sleep(0.5)
        
        final_screenshot_path = "final_state.png"
        self.take_screenshot(final_screenshot_path)
        time.sleep(1)
        self.cleanup_screenshot(final_screenshot_path)
        
        self.cleanup_all_screenshots()
        
        self.emit_status(f"Task completed: {results['steps_completed']}/{results['total_steps']} steps successful", "success")
        
        return results


class TaskExecutionThread(QThread):
    """Thread for executing tasks without blocking UI"""
    
    finished = pyqtSignal(dict)
    
    def __init__(self, agent, task):
        super().__init__()
        self.agent = agent
        self.task = task
    
    def run(self):
        result = self.agent.execute_task(self.task)
        self.finished.emit(result)


class PCControlUI(QMainWindow):
    """Main UI window for PC Control Agent"""
    
    def __init__(self):
        super().__init__()
        self.agent = None
        self.execution_thread = None
        self.init_ui()
        self.init_agent()
        self.position_window()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("PC Control AI Agent")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Title bar with close button
        title_bar = QFrame()
        title_bar.setStyleSheet("background-color: #2c3e50; padding: 5px;")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(5, 5, 5, 5)
        
        title_label = QLabel("PC Control AI Agent")
        title_label.setStyleSheet("color: white; font-weight: bold; font-size: 12px;")
        title_layout.addWidget(title_label)
        
        close_btn = QPushButton("×")
        close_btn.setFixedSize(25, 25)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        close_btn.clicked.connect(self.close)
        title_layout.addWidget(close_btn)
        
        main_layout.addWidget(title_bar)
        
        # Chat interface
        chat_frame = QFrame()
        chat_frame.setStyleSheet("background-color: #ecf0f1; border-radius: 5px;")
        chat_layout = QVBoxLayout(chat_frame)
        
        chat_label = QLabel("Chat Interface")
        chat_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #2c3e50;")
        chat_layout.addWidget(chat_label)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMaximumHeight(150)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
                font-size: 10px;
            }
        """)
        chat_layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        self.task_input = QLineEdit()
        self.task_input.setPlaceholderText("Enter your task here...")
        self.task_input.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 5px;
                font-size: 10px;
            }
        """)
        self.task_input.returnPressed.connect(self.execute_task)
        input_layout.addWidget(self.task_input)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.setFixedWidth(60)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 5px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.send_btn.clicked.connect(self.execute_task)
        input_layout.addWidget(self.send_btn)
        
        chat_layout.addLayout(input_layout)
        main_layout.addWidget(chat_frame)
        
        # Status display
        status_frame = QFrame()
        status_frame.setStyleSheet("background-color: #ecf0f1; border-radius: 5px; margin-top: 5px;")
        status_layout = QVBoxLayout(status_frame)
        
        status_label = QLabel("Live Status")
        status_label.setStyleSheet("font-weight: bold; font-size: 11px; color: #2c3e50;")
        status_layout.addWidget(status_label)
        
        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setMaximumHeight(200)
        self.status_display.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
                border-radius: 3px;
                padding: 5px;
                font-family: 'Courier New';
                font-size: 9px;
            }
        """)
        status_layout.addWidget(self.status_display)
        
        main_layout.addWidget(status_frame)
        
        # Set fixed size
        self.setFixedSize(400, 450)
        
        # Add welcome message
        self.add_chat_message("Welcome to PC Control AI Agent!", "system")
        self.add_chat_message("Enter a task and I'll help you complete it.", "system")
    
    def init_agent(self):
        """Initialize the PC Control Agent"""
        self.agent = PCControlAgent(status_callback=self.update_status)
        self.add_status("Agent initialized and ready", "success")
    
    def position_window(self):
        """Position window at bottom-right corner"""
        screen = QApplication.desktop().screenGeometry()
        x = screen.width() - self.width() - 20
        y = screen.height() - self.height() - 60
        self.move(x, y)
    
    def add_chat_message(self, message: str, sender: str = "user"):
        """Add a message to the chat display"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        if sender == "user":
            color = "#2c3e50"
            prefix = "You: "
        elif sender == "agent":
            color = "#27ae60"
            prefix = "Agent: "
        else:
            color = "#7f8c8d"
            prefix = "System: "
        
        cursor.insertHtml(f'<span style="color: {color}; font-weight: bold;">{prefix}</span>')
        cursor.insertHtml(f'<span style="color: #2c3e50;">{message}</span><br>')
        
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()
    
    def add_status(self, message: str, status_type: str = "info"):
        """Add a status message to the status display"""
        cursor = self.status_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        timestamp = time.strftime("%H:%M:%S")
        
        if status_type == "success":
            color = "#2ecc71"
            icon = "✓"
        elif status_type == "error":
            color = "#e74c3c"
            icon = "✗"
        elif status_type == "executing":
            color = "#f39c12"
            icon = "►"
        else:
            color = "#3498db"
            icon = "•"
        
        cursor.insertHtml(f'<span style="color: #95a5a6;">[{timestamp}]</span> ')
        cursor.insertHtml(f'<span style="color: {color};">{icon}</span> ')
        cursor.insertHtml(f'<span style="color: #ecf0f1;">{message}</span><br>')
        
        self.status_display.setTextCursor(cursor)
        self.status_display.ensureCursorVisible()
    
    def update_status(self, message: str, status_type: str = "info"):
        """Update status from agent callback"""
        self.add_status(message, status_type)
    
    def execute_task(self):
        """Execute the task entered by user"""
        task = self.task_input.text().strip()
        
        if not task:
            return
        
        if self.execution_thread and self.execution_thread.isRunning():
            self.add_chat_message("Please wait for the current task to complete.", "system")
            return
        
        self.add_chat_message(task, "user")
        self.task_input.clear()
        self.task_input.setEnabled(False)
        self.send_btn.setEnabled(False)
        
        # Start execution in separate thread
        self.execution_thread = TaskExecutionThread(self.agent, task)
        self.execution_thread.finished.connect(self.on_task_finished)
        self.execution_thread.start()
    
    def on_task_finished(self, result):
        """Handle task completion"""
        success = result.get("success", False)
        steps_completed = result.get("steps_completed", 0)
        total_steps = result.get("total_steps", 0)
        
        message = f"Task completed: {steps_completed}/{total_steps} steps successful"
        self.add_chat_message(message, "agent")
        
        self.task_input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.task_input.setFocus()
    
    def mousePressEvent(self, event):
        """Handle mouse press for window dragging"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for window dragging"""
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_position)
            event.accept()


def main():
    """Main function to run the PC Control Agent with UI"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = PCControlUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()