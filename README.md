# 🤖 OmniCtrl - Your Personal AI PC Assistant

Imagine having a personal assistant that can control your computer just by understanding your plain English commands. That's exactly what OmniCtrl does! Whether you're sitting at your desk or miles away from your computer, you can command it to do tasks for you through Telegram.

## 🌟 What Makes OmniCtrl Special?

- **🗣️ Understands Plain English**: Just say what you want, like "Open Chrome and search for cute cat videos"
- **👁️ Sees Your Screen**: Takes screenshots to understand what's currently on your screen
- **🖱️ Controls Your Mouse**: Automatically finds and clicks buttons, links, and menus
- **⌨️ Types For You**: Can type text in any application
- **🔄 Self-Healing**: If something goes wrong, it figures out what happened and tries a different approach
- **📱 Remote Control**: Control your PC from anywhere using Telegram on your phone
- **🎤 Voice Commands**: Send a voice message, and it will understand and execute your command
- **🔔 Notification Forwarding**: Get your PC notifications directly on your phone
- **⏰ Schedule Tasks**: Set tasks to run at specific times
- **📊 System Monitor**: Check your PC's CPU, RAM, and disk usage remotely
- **📹 Screen Streaming**: Record a video of your screen and send it to your phone
- **🎙️ Microphone Recording**: Record audio from your PC's microphone and send it to your phone
- **💬 Popup Messages**: Show important messages directly on your PC screen

## 🚀 How Does It Work?

1. You send a command via Telegram (text or voice)
2. The AI understands your command and creates a step-by-step plan
3. It takes screenshots to see what's currently on your screen
4. It identifies where to click or what to type
5. It performs all the actions automatically
6. If something goes wrong, it analyzes the situation and tries a different approach
7. You get a summary of what was done, along with a final screenshot

## 📋 Requirements

Before you begin, make sure you have:
- A Windows 10 or Windows 11 computer
- An internet connection
- At least 2GB of free RAM
- A Telegram account on your phone

## 🔧 Installation Guide (Step-by-Step for Beginners)

### Step 1: Install Python

Python is a programming language that OmniCtrl needs to run.

1. Open your web browser and go to [python.org](https://www.python.org/downloads/)
2. Click the big yellow button that says "Download Python" (it will show the latest version)
3. Once the download is complete, open the downloaded file
4. **IMPORTANT**: At the bottom of the installation window, make sure to check the box that says "Add Python to PATH" (this is crucial!)
5. Click "Install Now"
6. Wait for the installation to complete
7. Click "Close" when done

### Step 2: Install Tesseract OCR (For Reading Screen Text)

Tesseract helps OmniCtrl read text from your screen.

1. Download Tesseract from this link: [Tesseract OCR for Windows](https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe)
2. Open the downloaded file
3. Follow the installation wizard (just keep clicking "Next")
4. When asked, make sure to check the box that says "Add Tesseract to system PATH"
5. Click "Install" and wait for it to finish
6. Click "Finish" to close the installer

### Step 3: Download the OmniCtrl Project

Now let's get the actual OmniCtrl program.

1. Press the Windows key on your keyboard
2. Type "cmd" and press Enter (this opens the Command Prompt)
3. In the black window that appears, type or copy-paste this command and press Enter:
   ```
   git clone https://github.com/Aryan-crypt/Computer-Agent.git
   ```
4. Wait for the download to complete
5. Now type this command and press Enter to go into the project folder:
   ```
   cd Computer-Agent
   ```

### Step 4: Install Required Python Packages

OmniCtrl needs some additional components to work properly.

1. In the same Command Prompt window, type or copy-paste this command and press Enter:
   ```
   pip install -r requirements.txt
   ```
2. Wait for all the packages to install (this might take a few minutes)
3. When it's done, you'll see a message saying "Successfully installed..."

### Step 5: Get Your API Keys

OmniCtrl uses AI services that require API keys. Don't worry, these are mostly free to use!

#### Getting a Gemini API Key:
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Choose a Google Cloud project (or create a new one)
5. Copy the API key that appears (it will look something like "AIzaSy...")
6. Keep this key safe, you'll need it soon

#### Getting an OpenRouter API Key:
1. Go to [OpenRouter](https://openrouter.ai/keys)
2. Click "Sign In" and create an account
3. Once logged in, click "Create Key"
4. Give it a name (like "OmniCtrl") and click "Create"
5. Copy the API key that appears
6. Keep this key safe too

#### Creating a Telegram Bot:
1. Open Telegram on your phone
2. Search for "BotFather" (it has a blue checkmark)
3. Send the message "/newbot"
4. Follow the instructions to name your bot
5. You'll receive a message with your bot token (it will look something like "123456:ABC-DEF...")
6. Copy this token and keep it safe

#### Finding Your Telegram User ID:
1. Search for "userinfobot" on Telegram
2. Send it any message
3. It will reply with your user ID (a number like 123456789)
4. Note down this number

### Step 6: Configure OmniCtrl

Now let's put all those keys in the right place.

1. In the File Explorer, navigate to the Computer-Agent folder you downloaded
2. Find and open the file named "API.py" (right-click and choose "Open with" > "Notepad")
3. You'll see something like this:
   ```python
   GEMINI_API_KEY = "YOUR_API_KEY"
   OpenRouter_API_KEY = "YOUR_API_KEY"
   TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
   AUTHORIZED_USERS = [123,123]
   ```
4. Replace "YOUR_API_KEY" next to GEMINI_API_KEY with the Gemini key you copied
5. Replace "YOUR_API_KEY" next to OpenRouter_API_KEY with the OpenRouter key you copied
6. Replace "YOUR_BOT_TOKEN" with the Telegram bot token you received
7. Replace the numbers [123,123] with your Telegram user ID in square brackets, like [123456789]
8. Save the file (Ctrl+S) and close Notepad

## 🎮 How to Use OmniCtrl

### Method 1: Remote Control via Telegram (Recommended)

1. Open Command Prompt
2. Navigate to the Computer-Agent folder:
   ```
   cd Computer-Agent
   ```
3. Run the Telegram interface with this command:
   ```
   python main.py
   ```
4. You'll see messages indicating the bot is connecting to Telegram
5. Open Telegram on your phone
6. Find the bot you created and start a conversation with it
7. Send any command as a text message or voice note
8. The bot will confirm before executing your command
9. You'll receive updates and a final screenshot when it's done

### Method 2: Running Locally (At Your Computer)

1. Open Command Prompt
2. Navigate to the Computer-Agent folder (if you're not already there):
   ```
   cd Computer-Agent
   ```
3. Run the program with this command:
   ```
   python Core/core_agent.py
   ```
4. You'll see a prompt asking for your task
5. Type what you want the agent to do (e.g., "Open Chrome and search for weather today")
6. Confirm by typing "y" and pressing Enter
7. Watch the agent work its magic!

## 💡 Example Commands to Try

Here are some things you can ask OmniCtrl to do:

- "Open Chrome browser and search for 'what are allotropes of carbon' then write it in notepad"
- "Open calculator and calculate 25 * 37"
- "Take a screenshot and save it to desktop"
- "Open notepad and write a short poem about computers"
- "Open File Explorer and navigate to Documents folder"
- "Open Spotify and play your liked songs"
- "Check the weather for today"
- "Create a new folder on my desktop called 'Projects'"
- "Open Word and write a letter to my friend"

## 📱 Telegram Commands Reference

Once you're connected via Telegram, you can use these special commands:

| Command | Description |
|---------|-------------|
| `/start` | Start the bot and see available features |
| `/help` | Display full help guide |
| `/task <text>` | Explicitly execute a task |
| `/status` | Check if the agent is busy or idle |
| `/screenshot` | Take a screenshot of your PC |
| `/stop` | Emergency stop for running tasks |
| `/history` | See your recent task history |
| `/clear` | Clear task history |
| `/sysinfo` | Check CPU, RAM, and disk usage |
| `/clip` | Read your PC's clipboard content |
| `/clip set <text>` | Write text to your PC's clipboard |
| `/webcam` | Take a photo with your webcam |
| `/getfile <name>` | Get a file from your Downloads folder |
| `/stream` | Record a 10-second video of your screen |
| `/mic` | Record 10 seconds of microphone audio |
| `/popup <message>` | Show a message popup on your PC |
| `/alias add <word> <command>` | Create a shortcut for a command |
| `/alias list` | List all your shortcuts |
| `/alias del <word>` | Delete a shortcut |
| `/schedule HH:MM <task>` | Schedule a task for a specific time |
| `/schedule list` | List all scheduled tasks |
| `/schedule clear` | Clear all scheduled tasks |

You can also:
- Send a **voice message** to have it transcribed and executed
- Send a **file** to save it directly to your Downloads folder

## ⚠️ Safety Features

- **Emergency Stop**: Send `/stop` in Telegram or move your mouse to the top-left corner of your screen to immediately stop all actions
- **Confirmation Required**: The agent always asks for confirmation before executing tasks
- **Authorization**: Only users you specify can control your PC
- **Error Handling**: If something goes wrong, the agent tries to recover or stops safely
- **Rate Limiting**: Prevents abuse by limiting how many commands can be sent in a minute
- **Task Timeout**: Tasks automatically stop if they take too long (5 minutes by default)

## 🛠️ Troubleshooting Common Issues

### "Python is not recognized" Error
- This means Python wasn't added to your PATH during installation
- Uninstall Python, reinstall it, and make sure to check "Add Python to PATH"

### "Tesseract is not installed" Error
- Make sure you installed Tesseract correctly
- Try restarting your computer after installation
- Check if Tesseract is in your system PATH

### API Key Errors
- Double-check that your API keys are correct
- Make sure you didn't include any extra spaces or quotes
- Verify your internet connection is stable

### Bot Not Responding in Telegram
- Check if the bot token is correct
- Make sure your Telegram user ID is added to AUTHORIZED_USERS
- Verify the program is still running in Command Prompt

### "ModuleNotFoundError" Errors
- Some required packages might not have installed correctly
- Try running `pip install -r requirements.txt` again
- Look for any error messages during installation

### Screen Recording/Mic Recording Not Working
- Make sure you have the necessary dependencies installed
- Try running `pip install sounddevice soundfile numpy`
- Check if your microphone/webcam is properly connected

## 🔄 Updating OmniCtrl

To get the latest features and fixes:

1. Open Command Prompt
2. Navigate to the Computer-Agent folder:
   ```
   cd Computer-Agent
   ```
3. Run these commands:
   ```
   git pull
   pip install -r requirements.txt
   ```
4. Restart the program

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository on GitHub
2. Create a new branch for your feature
3. Make your changes
4. Test everything thoroughly
5. Submit a pull request

For major changes, please open an issue first to discuss what you would like to change.

## 🙏 Acknowledgments

- [Gemini 3.0 Flash](https://ai.google.dev/) for task planning and understanding
- [Gemma 4](https://openrouter.ai/) for coordinate extraction and content processing
- [python-telegram-bot](https://python-telegram-bot.org/) for the Telegram interface
- The open-source community for the various libraries used

## ⚠️ Disclaimer

This tool automates interactions with your computer. While safety features are in place, use it at your own risk. The developers are not responsible for any unintended actions or data loss. Always review the planned steps before confirming execution.

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Search existing issues on [GitHub](https://github.com/Aryan-crypt/Computer-Agent/issues)
3. Create a new issue with details about your problem
4. Include screenshots of any error messages

---

**Enjoy your personal AI PC assistant!** 🎉

*If you find this project useful, please consider giving it a star on GitHub!*