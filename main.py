"""
PC Control AI Agent with Multi-Model Architecture
Uses Gemini 3.0 Flash for planning, Gemma 4 (via OpenRouter) for coordinate extraction
and content reading/writing

NEW FEATURE: Self-healing Re-plan approach on step failure.
"""

import os
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
import google.genai as genai
from google.genai import types as genai_types
import requests
import urllib.parse
from API import *

# Configure pyautogui for safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

# Configuration
OPENROUTER_API_KEY = OpenRouter_API_KEY
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMMA4_MODEL = "google/gemma-4-31b-it:free"

# Configure Gemini
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

@dataclass
class Step:
    """Represents a single step in the task execution"""
    description: str
    action_type: str  # 'keyboard', 'mouse_click', 'mouse_move', 'type_text', 'read_content'
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
    """Gemini 3.0 Flash planning and command agent"""
    
    def __init__(self):
        self.model = "gemini-3-flash-preview"
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
            - Calculator can be used through Keyboard (e.g; using numbers (1,2,3,4,5,6,7,8,9,0), division (/), multiplication (*), addition (+), subtraction (-) and results or equals to (=)).
            - For enabling the searchbar in youtube click '/' then type the search query.
            - Notepad can be opened by :- clicking "windows + R" then typing "notepad" and then clicking "enter" button.
        6. Remember :
            - Do not copy or select any text, because you are not allowed to select text, rather than read_content then type_text.
            - Always WAIT for an website to load before proceeding to the next step.

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
            contents = [system_prompt]
            if current_screenshot:
                img_byte_arr = BytesIO()
                current_screenshot.save(img_byte_arr, format='PNG')
                contents.append(
                    genai_types.Part.from_bytes(
                        data=img_byte_arr.getvalue(),
                        mime_type="image/png"
                    )
                )

            response = gemini_client.models.generate_content(
                model=self.model,
                contents=contents
            )
            
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

    def replan_task(self, user_prompt: str, failed_step: Step, remaining_steps: List[Step], current_screenshot: Image.Image) -> List[Step]:
        """Generate a NEW recovery plan after a step fails"""
        
        # Convert remaining steps to JSON string so the AI knows what it hasn't done yet
        remaining_steps_str = json.dumps([s.__dict__ for s in remaining_steps], indent=2)
        
        system_prompt = f"""You are an AI assistant controlling a Windows 11 PC. 
        We are trying to complete this task: "{user_prompt}"
        
        PROBLEM: We just FAILED to execute this step:
        - Description: {failed_step.description}
        - Action Type: {failed_step.action_type}
        - Parameters: {json.dumps(failed_step.parameters)}
        
        Look at the attached screenshot. The screen is currently in this state AFTER the failure.
        These were the steps we HAD LEFT to do:
        {remaining_steps_str}
        
        INSTRUCTIONS:
        1. Look at the screenshot. Did the failure leave us in an unexpected state? (e.g., a popup, wrong window)
        2. Do NOT just repeat the exact same failed step unless you are 100% sure it will work now.
        3. Try an alternative approach (e.g., use a different keyboard shortcut, click somewhere else to close a popup first).
        4. Generate a NEW JSON array of steps starting from recovering from this failure, all the way to finishing the original task.
        
        Return ONLY the JSON array of steps using the exact same format as before.
        """
        
        try:
            contents = [system_prompt]
            img_byte_arr = BytesIO()
            current_screenshot.save(img_byte_arr, format='PNG')
            contents.append(
                genai_types.Part.from_bytes(
                    data=img_byte_arr.getvalue(),
                    mime_type="image/png"
                )
            )

            response = gemini_client.models.generate_content(
                model=self.model,
                contents=contents
            )
            
            response_text = response.text
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                steps_data = json.loads(json_match.group())
                return [Step(**step) for step in steps_data]
            else:
                print(f"Could not parse RE-plan from response: {response_text}")
                return []
                
        except Exception as e:
            print(f"Error in RE-planning: {e}")
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

            response = gemini_client.models.generate_content(
                model=self.model,
                contents=[
                    prompt,
                    genai_types.Part.from_bytes(
                        data=img_byte_arr.getvalue(),
                        mime_type="image/png"
                    )
                ]
            )
            
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
    """Gemma 4 (via OpenRouter) for extracting coordinates from screenshots"""
    
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.base_url = OPENROUTER_BASE_URL
        self.model = GEMMA4_MODEL
    
    def extract_coordinates(self, screenshot_path: str, target_description: str) -> Optional[Tuple[int, int]]:
        """Extract coordinates of target element from screenshot using Gemma 4"""
        
        try:
            width, height = pyautogui.size()

            with open(screenshot_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            data_uri = f"data:image/png;base64,{base64_image}"

            prompt = (
                f"You are a UI assistant. Look at the screenshot and locate: {target_description}. "
                "Return ONLY a JSON object with keys 'ymin', 'xmin', 'ymax', 'xmax' using a 0-1000 coordinate system. "
                "No explanation, no markdown, just the JSON."
            )

            response = requests.post(
                url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_uri}}
                            ]
                        }
                    ]
                },
                timeout=30
            )

            response.raise_for_status()
            text_response = response.json()["choices"][0]["message"]["content"]
            print(f"[DEBUG] Gemma 4 coordinate response: {text_response}")

            # Parse coordinates — try JSON first, fallback to regex
            try:
                obj = json.loads(text_response)
                ymin, xmin, ymax, xmax = [float(obj[k]) for k in ("ymin", "xmin", "ymax", "xmax")]
            except Exception:
                numbers = re.findall(r"(\d+(?:\.\d+)?)", text_response)
                if len(numbers) >= 4:
                    ymin, xmin, ymax, xmax = [float(n) for n in numbers[:4]]
                else:
                    raise ValueError("Could not find 4 coordinates in response.")

            # Convert 0-1000 scale to actual screen pixels
            center_x_norm = (xmin + xmax) / 2
            center_y_norm = (ymin + ymax) / 2
            pixel_x = int((center_x_norm / 1000) * width)
            pixel_y = int((center_y_norm / 1000) * height)

            print(f"Target found at: ({pixel_x}, {pixel_y})")
            return (pixel_x, pixel_y)

        except Exception as e:
            print(f"Error extracting coordinates with Gemma 4 (OpenRouter): {e}")
            print("Trying Gemini 3 Flash as backup for coordinate extraction...")
            return self._gemini_fallback_extraction(screenshot_path, target_description)

    def _gemini_fallback_extraction(self, screenshot_path: str, target_description: str) -> Optional[Tuple[int, int]]:
        """Fallback: use Gemini 3 Flash to extract coordinates"""
        try:
            width, height = pyautogui.size()
            with open(screenshot_path, "rb") as f:
                img_bytes = f.read()
            prompt = (
                f"Detect the {target_description}. "
                "Return only the coordinates as [ymin, xmin, ymax, xmax] "
                "using a 0-1000 coordinate system. No explanation, just the numbers."
            )
            response = gemini_client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[
                    prompt,
                    genai_types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                ]
            )
            text_response = response.text
            print(f"[DEBUG] Gemini 3 Flash coordinate response: {text_response}")

            numbers = re.findall(r"(\d+(?:\.\d+)?)", text_response)
            if len(numbers) >= 4:
                ymin, xmin, ymax, xmax = [float(n) for n in numbers[:4]]
                center_x_norm = (xmin + xmax) / 2
                center_y_norm = (ymin + ymax) / 2
                pixel_x = int((center_x_norm / 1000) * width)
                pixel_y = int((center_y_norm / 1000) * height)
                print(f"[Gemini Fallback] Target found at: ({pixel_x}, {pixel_y})")
                return (pixel_x, pixel_y)
            else:
                raise ValueError(f"Could not find 4 coordinates in Gemini response: {text_response}")

        except Exception as e:
            print(f"Error extracting coordinates with Gemini 3 Flash: {e}")
            return self._center_fallback(screenshot_path)

    def _center_fallback(self, screenshot_path: str) -> Optional[Tuple[int, int]]:
        """Last resort: return center of screen"""
        try:
            img = Image.open(screenshot_path)
            print("[Fallback] Using center of screen as coordinate.")
            return (img.width // 2, img.height // 2)
        except:
            return None

class ContentProcessor:
    """Gemma 4 (via OpenRouter) for reading and writing content"""
    
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.base_url = OPENROUTER_BASE_URL
        self.model = GEMMA4_MODEL

    def _call_gemma4(self, messages: list, max_tokens: int = 1000) -> str:
        """Helper to call Gemma 4 via OpenRouter"""
        response = requests.post(
            url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _call_gemini_flash(self, prompt: str, screenshot: Image.Image = None) -> str:
        """Fallback: call Gemini 3 Flash for content reading/generation"""
        contents = [prompt]
        if screenshot:
            img_byte_arr = BytesIO()
            screenshot.save(img_byte_arr, format="PNG")
            contents.append(
                genai_types.Part.from_bytes(
                    data=img_byte_arr.getvalue(),
                    mime_type="image/png"
                )
            )
        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=contents
        )
        return response.text

    def read_content(self, screenshot: Image.Image, instruction: str = "Read all text from this image") -> str:
        """Read and extract content from screenshot using Gemma 4, with Gemini 3 Flash backup"""
        buffered = BytesIO()
        screenshot.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        data_uri = f"data:image/png;base64,{img_base64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }
        ]
        
        try:
            return self._call_gemma4(messages, max_tokens=1000)
        except Exception as e:
            print(f"Error reading content with Gemma 4 (OpenRouter): {e}")
            print("Trying Gemini 3 Flash as backup for content reading...")
            try:
                return self._call_gemini_flash(instruction, screenshot)
            except Exception as e2:
                print(f"Error reading content with Gemini 3 Flash: {e2}")
                return self._ocr_fallback(screenshot)

    def _ocr_fallback(self, screenshot: Image.Image) -> str:
        """Last resort OCR fallback"""
        try:
            text = pytesseract.image_to_string(screenshot)
            return text.strip()
        except Exception as e:
            print(f"OCR fallback failed: {e}")
            return ""
    
    def generate_text(self, prompt: str) -> str:
        """Generate text using Gemma 4, with Gemini 3 Flash backup"""
        messages = [{"role": "user", "content": prompt}]
        try:
            return self._call_gemma4(messages, max_tokens=500)
        except Exception as e:
            print(f"Error generating text with Gemma 4 (OpenRouter): {e}")
            print("Trying Gemini 3 Flash as backup for text generation...")
            try:
                return self._call_gemini_flash(prompt)
            except Exception as e2:
                print(f"Error generating text with Gemini 3 Flash: {e2}")
                return ""

class PCControlAgent:
    """Main orchestrator agent that coordinates all components"""
    
    def __init__(self):
        self.commander = CommanderAgent()
        self.coordinate_extractor = CoordinateExtractor()
        self.content_processor = ContentProcessor()
        self.current_screenshot = None
        self.screenshot_history = []
        self.temp_screenshots = []  # Track temporary screenshot files
        
        # Set up pyautogui settings
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.5
        
    def take_screenshot(self, save_path: Optional[str] = None) -> Image.Image:
        """Take a screenshot of the entire screen"""
        screenshot = pyautogui.screenshot()
        self.current_screenshot = screenshot
        self.screenshot_history.append(screenshot)
        
        if save_path:
            screenshot.save(save_path)
            self.temp_screenshots.append(save_path)  # Track file for cleanup
            
        # Keep only last 10 screenshots in memory
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
            pass # Silent cleanup
    
    def cleanup_all_screenshots(self):
        """Delete all temporary screenshots"""
        for filepath in self.temp_screenshots[:]:  # Create copy to iterate
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
        
        print(f"\nExecuting step: {step.description}")
        
        temp_screenshot_path = None
        
        try:
            if step.action_type == "keyboard":
                keys = step.parameters.get("keys", [])
                success = self.execute_keyboard_action(keys)
                
            elif step.action_type == "mouse_click":
                # Take screenshot and find coordinates
                temp_screenshot_path = "temp_screenshot.png"
                screenshot = self.take_screenshot(temp_screenshot_path)
                target = step.parameters.get("target", "")
                
                # Try to extract coordinates
                coords = self.coordinate_extractor.extract_coordinates(temp_screenshot_path, target)
                
                if coords:
                    success = self.execute_mouse_click(coords[0], coords[1])
                else:
                    print(f"Could not find coordinates for: {target}")
                    # Try alternative approach with Commander analysis
                    analysis = self.commander.analyze_screenshot(screenshot, target)
                    if analysis.get("use_keyboard") and analysis.get("keyboard_shortcut"):
                        success = self.execute_keyboard_action(analysis["keyboard_shortcut"])
                    else:
                        success = False
                
                # Clean up temp screenshot immediately after use
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
                print(f"Read content: {content[:200]}...")
                success = True
                
            elif step.action_type == "wait":
                duration = step.parameters.get("duration", 1.0)
                time.sleep(duration)
                success = True
                
            else:
                print(f"Unknown action type: {step.action_type}")
                success = False
                
            step.completed = success
            return success
            
        except Exception as e:
            print(f"Error executing step: {e}")
            # Clean up on error
            if temp_screenshot_path:
                self.cleanup_screenshot(temp_screenshot_path)
            step.completed = False
            return False
    
    def execute_task(self, user_prompt: str) -> Dict[str, Any]:
        """Main method to execute a complete task with Self-Healing Re-plan capability"""
        
        print(f"\n{'='*60}")
        print(f"Starting task: {user_prompt}")
        print(f"{'='*60}")
        
        # SAFETY CONFIG: Maximum times the AI is allowed to re-plan before giving up
        MAX_REPLAN_ATTEMPTS = 3 
        replan_count = 0
        
        # First, minimize all windows
        print("\nMinimizing all windows...")
        self.execute_keyboard_action(["win", "m"])
        time.sleep(1)
        
        # Take initial screenshot
        initial_screenshot = self.take_screenshot("initial_state.png")
        
        # Generate plan
        print("\nGenerating execution plan...")
        steps = self.commander.plan_task(user_prompt, initial_screenshot)
        
        # Clean up initial screenshot after plan is generated
        self.cleanup_screenshot("initial_state.png")
        
        if not steps:
            print("Failed to generate plan")
            return {"success": False, "error": "Could not generate execution plan",
                    "steps_completed": 0, "total_steps": 0, "step_results": [], "replans": 0}
        
        print(f"\nPlan generated with {len(steps)} steps:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step.description}")
        
        # Execute steps using a WHILE loop so we can dynamically swap steps if we re-plan
        results = {
            "success": True,
            "steps_completed": 0,
            "total_steps": len(steps),
            "step_results": [],
            "replans": 0
        }
        
        current_step_idx = 0
        
        while current_step_idx < len(steps):
            step = steps[current_step_idx]
            
            # Update total steps display in case replanning added steps
            results["total_steps"] = len(steps)
            
            print(f"\n[Step {current_step_idx + 1}/{len(steps)}]")
            
            # Take screenshot before each step
            step_screenshot_path = f"step_{current_step_idx}_before.png"
            self.take_screenshot(step_screenshot_path)
            
            # Execute the step
            success = self.execute_step(step)
            
            # Clean up step screenshot after execution
            self.cleanup_screenshot(step_screenshot_path)
            
            results["step_results"].append({
                "step": current_step_idx + 1,
                "description": step.description,
                "success": success,
                "result": step.result
            })
            
            if success:
                results["steps_completed"] += 1
                current_step_idx += 1  # Move to next step
            else:
                print(f"\n❌ FAILURE DETECTED on: {step.description}")
                
                # CHECK SAFETY LIMIT
                if replan_count >= MAX_REPLAN_ATTEMPTS:
                    print(f"🛑 SAFETY LIMIT REACHED: Agent has re-planned {MAX_REPLAN_ATTEMPTS} times. Aborting task to prevent infinite loops.")
                    results["success"] = False
                    break
                
                print(f"🔄 Initiating Re-Plan ({replan_count + 1}/{MAX_REPLAN_ATTEMPTS})...")
                replan_count += 1
                results["replans"] = replan_count
                time.sleep(1) # Give PC a second to settle after the error
                
                # Take fresh screenshot of the "broken" state
                replan_screenshot_path = "replan_state.png"
                current_state_screenshot = self.take_screenshot(replan_screenshot_path)
                
                # Get the steps that HAVEN'T been executed yet
                remaining_steps = steps[current_step_idx:]
                
                # Ask AI to figure out what went wrong and create a new plan
                print("Asking AI to analyze failure and generate recovery steps...")
                new_recovery_steps = self.commander.replan_task(
                    user_prompt=user_prompt,
                    failed_step=step,
                    remaining_steps=remaining_steps,
                    current_screenshot=current_state_screenshot
                )
                
                self.cleanup_screenshot(replan_screenshot_path)
                
                if new_recovery_steps:
                    print(f"✅ Successfully generated {len(new_recovery_steps)} recovery steps!")
                    for i, s in enumerate(new_recovery_steps, 1):
                        print(f"  {i}. {s.description}")
                    
                    # MAGIC HAPPENS HERE: 
                    # We keep the successful past steps, and replace everything from current index onwards with the new plan
                    steps = steps[:current_step_idx] + new_recovery_steps
                else:
                    print("❌ AI failed to generate a recovery plan. Aborting task.")
                    results["success"] = False
                    break
            
            # Small delay between steps
            time.sleep(0.5)
        
        # Final cleanup of any remaining screenshots
        self.cleanup_all_screenshots()
        
        print(f"\n{'='*60}")
        if results["success"]:
             print(f"Task completed successfully!")
        else:
             print(f"Task aborted/failed.")
             
        print(f"Steps completed: {results['steps_completed']}/{results['total_steps']}")
        print(f"Total Re-plans used: {results['replans']}")
        print(f"{'='*60}")
        
        return results

def main():
    """Main function to run the PC Control Agent"""
    
    print("PC Control AI Agent (WITH SELF-HEALING RE-PLAN)")
    print("=" * 60)
    print("This agent can control your PC to complete tasks.")
    print("If a step fails, it will look at the screen and try a new approach!")
    print("Type 'exit' to quit.")
    print("=" * 60)
    
    # Initialize the agent
    agent = PCControlAgent()
    
    # Example tasks
    example_tasks = [
        "Open Chrome browser and search for 'what are allotropes of carbon' then write it in notepad",
        "Open calculator and calculate 25 * 37",
        "Open notepad and write a short poem about computers"
    ]
    
    print("\nExample tasks you can try:")
    for i, task in enumerate(example_tasks, 1):
        print(f"  {i}. {task}")
    
    try:
        while True:
            print("\n" + "-" * 60)
            user_input = input("Enter your task (or 'exit' to quit): ").strip()
            
            if user_input.lower() == 'exit':
                print("Exiting PC Control Agent. Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Safety confirmation
            print(f"\nYou want me to: {user_input}")
            confirm = input("Proceed? (y/n): ").strip().lower()
            
            if confirm != 'y':
                print("Task cancelled.")
                continue
            
            try:
                # Execute the task
                results = agent.execute_task(user_input)
                
                # Print summary
                print("\n" + "=" * 60)
                print("TASK SUMMARY")
                print("=" * 60)
                print(f"Task: {user_input}")
                print(f"Final Status: {'SUCCESS' if results['success'] else 'FAILED/ABORTED'}")
                print(f"Steps completed: {results['steps_completed']}/{results['total_steps']}")
                print(f"Times AI had to Re-plan: {results.get('replans', 0)}")
                
                if results.get('step_results'):
                    print("\nStep Details:")
                    for step_result in results['step_results']:
                        status = "✓" if step_result['success'] else "✗"
                        print(f"  {status} Step {step_result['step']}: {step_result['description']}")
                        if step_result.get('result'):
                            print(f"    Result: {step_result['result'][:100]}...")
                
            except KeyboardInterrupt:
                print("\nTask interrupted by user.")
                # Clean up on interrupt
                agent.cleanup_all_screenshots()
            except Exception as e:
                print(f"\nError during task execution: {e}")
                import traceback
                traceback.print_exc()
                # Clean up on error
                agent.cleanup_all_screenshots()
    
    finally:
        # Final cleanup when exiting the program
        print("\nPerforming final cleanup...")
        agent.cleanup_all_screenshots()

if __name__ == "__main__":
    main()