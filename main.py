"""
PC Control AI Agent with Multi-Model Architecture
Uses Gemini 2.5 Pro for planning, Holo 1.5 for coordinate extraction, 
and Pollinations OpenAI for content reading/writing
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
import google.generativeai as genai
from gradio_client import Client, handle_file
import requests
import urllib.parse
from API import GEMINI_API_KEY

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
        1. Always start by minimizing all windows (Win+M) for a clean slate
        2. Break down the task into small, atomic steps
        3. Prefer keyboard shortcuts when possible
        4. Always before typing on an input area, click on that input area before typing
        5. For each step, specify if it needs:
           - Keyboard shortcut
           - Mouse click (will need coordinate extraction)
           - Text input
           - Content reading from screen
        6. Tips :
            - Claculator can be used through Keyboard (e.g; using numbers (1,2,3,4,5,6,7,8,9,0), division (/), multiplication (*), addition (+), subtraction (-) and results or equals to (=)).
            - For enabling the searchbar in youtube click '/' then type the search query.
        
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
                # Convert PIL Image to bytes for Gemini
                img_byte_arr = BytesIO()
                current_screenshot.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                response = self.model.generate_content([
                    system_prompt,
                    {"mime_type": "image/png", "data": img_byte_arr}
                ])
            else:
                response = self.model.generate_content(system_prompt)
            
            # Parse JSON response
            response_text = response.text
            # Extract JSON from response
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
            # Fallback to center of screen if Holo is not available
            print("Holo 1.5 not available, using fallback coordinate extraction")
            return self._fallback_extraction(screenshot_path, target_description)
        
        try:
            result = self.client.predict(
                input_numpy_image=handle_file(screenshot_path),
                task=f"Find and click on: {target_description}",
                api_name="/localize"
            )
            
            # Parse coordinates from result
            # Result format may vary, looking for x,y coordinates
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
            # Simple center-based fallback
            return (img.width // 2, img.height // 2)
        except:
            return None

class ContentProcessor:
    """Pollinations OpenAI model for reading and writing content"""
    
    def __init__(self):
        #self.api_key = POLLINATIONS_API_KEY
        self.base_url = POLLINATIONS_BASE_URL
        
    def read_content(self, screenshot: Image.Image, instruction: str = "Read all text from this image") -> str:
        """Read and extract content from screenshot"""
        
        # Convert image to base64
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
            # Fallback to OCR
            return self._ocr_fallback(screenshot)
    
    def _ocr_fallback(self, screenshot: Image.Image) -> str:
        """Fallback OCR method"""
        try:
            text = pytesseract.image_to_string(screenshot)
            return text.strip()
        except Exception as e:
            print(f"OCR fallback failed: {e}")
            return ""
    
    def generate_text(self, prompt: str) -> str:
        """Generate text based on prompt"""
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": "openai",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error generating text: {e}")
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
                print(f"Deleted screenshot: {filepath}")
                if filepath in self.temp_screenshots:
                    self.temp_screenshots.remove(filepath)
        except Exception as e:
            print(f"Warning: Could not delete {filepath}: {e}")
    
    def cleanup_all_screenshots(self):
        """Delete all temporary screenshots"""
        print("\nCleaning up temporary screenshots...")
        for filepath in self.temp_screenshots[:]:  # Create copy to iterate
            self.cleanup_screenshot(filepath)
        self.temp_screenshots.clear()
        print("Screenshot cleanup complete.")
    
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
        """Main method to execute a complete task"""
        
        print(f"\n{'='*60}")
        print(f"Starting task: {user_prompt}")
        print(f"{'='*60}")
        
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
            return {"success": False, "error": "Could not generate execution plan"}
        
        print(f"\nPlan generated with {len(steps)} steps:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step.description}")
        
        # Execute steps
        results = {
            "success": True,
            "steps_completed": 0,
            "total_steps": len(steps),
            "step_results": []
        }
        
        for i, step in enumerate(steps, 1):
            print(f"\n[Step {i}/{len(steps)}]")
            
            # Take screenshot before each step
            step_screenshot_path = f"step_{i}_before.png"
            self.take_screenshot(step_screenshot_path)
            
            # Execute the step
            success = self.execute_step(step)
            
            # Clean up step screenshot after execution
            self.cleanup_screenshot(step_screenshot_path)
            
            results["step_results"].append({
                "step": i,
                "description": step.description,
                "success": success,
                "result": step.result
            })
            
            if success:
                results["steps_completed"] += 1
            else:
                print(f"Step {i} failed. Attempting to continue...")
                # Optionally, we could try to recover here
            
            # Small delay between steps
            time.sleep(0.5)
        
        # Take final screenshot
        final_screenshot_path = "final_state.png"
        self.take_screenshot(final_screenshot_path)
        
        # Clean up final screenshot after a short delay (to allow viewing if needed)
        time.sleep(1)
        self.cleanup_screenshot(final_screenshot_path)
        
        # Final cleanup of any remaining screenshots
        self.cleanup_all_screenshots()
        
        print(f"\n{'='*60}")
        print(f"Task completed: {results['steps_completed']}/{results['total_steps']} steps successful")
        print(f"{'='*60}")
        
        return results

def main():
    """Main function to run the PC Control Agent"""
    
    print("PC Control AI Agent")
    print("=" * 60)
    print("This agent can control your PC to complete tasks.")
    print("Type 'exit' to quit.")
    print("=" * 60)
    
    # Initialize the agent
    agent = PCControlAgent()
    
    # Example tasks
    example_tasks = [
        "Open Chrome browser and search for 'what are allotropes of carbon' then write it in notepad",
        "Open calculator and calculate 25 * 37",
        "Take a screenshot and save it to desktop",
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
                print(f"Success: {results['success']}")
                print(f"Steps completed: {results['steps_completed']}/{results['total_steps']}")
                
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