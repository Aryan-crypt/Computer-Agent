

# PC Control AI Agent

A smart assistant that can control your Windows computer using natural language commands. Just tell it what you want to do, and it will break down the task and perform all the necessary actions on your computer automatically!

## 🌟 What Can It Do?

- **Understand Your Commands**: Simply type what you want to do in plain English
- **Automate Tasks**: Open applications, search the web, write documents, and more
- **See What You See**: Takes screenshots to understand what's on your screen
- **Click Where Needed**: Automatically finds and clicks buttons, links, and menus
- **Read Screen Content**: Can read text from your screen using advanced OCR technology
- **Shows What It's Doing**: A special debug window shows you exactly what the agent is doing in real-time

## 🚀 How It Works

1. You tell the agent what you want to do (like "Open Chrome and search for cats")
2. The AI creates a step-by-step plan to complete your task
3. It takes screenshots to see what's currently on your screen
4. It identifies where to click or what to type
5. It performs all the actions automatically
6. You can watch everything happening in the debug window

## 📋 Requirements

- Windows 10 or Windows 11
- Python 3.8 or newer
- Internet connection
- At least 2GB of free RAM
- A mouse and keyboard

## 🔧 Installation Guide

Follow these steps carefully to get everything working:

### Step 1: Install Python

1. Go to [python.org](https://www.python.org/downloads/)
2. Download the latest Python version for Windows
3. Run the installer
4. **IMPORTANT**: Check the box that says "Add Python to PATH"
5. Click "Install Now"

### Step 2: Install Tesseract OCR (Required for reading text from screen)

1. Download Tesseract from this link: [Tesseract OCR for Windows](https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe)
2. Run the downloaded file
3. Follow the installation wizard
4. **IMPORTANT**: Remember the installation path (usually `C:\Program Files\Tesseract-OCR`)
5. When asked, check the box to add Tesseract to your system PATH

### Step 3: Download the Project

1. Go to the GitHub repository
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file to a folder on your computer

### Step 4: Set Up the Project

1. Open the Start Menu and type "cmd"
2. Right-click on "Command Prompt" and select "Run as administrator"
3. Navigate to the project folder using the `cd` command
   - Example: `cd C:\Users\YourName\Downloads\pc-control-ai-agent`
4. Create a virtual environment:
   ```
   python -m venv .venv
   ```
5. Activate the virtual environment:
   ```
   .venv\Scripts\activate
   ```
6. Install all required packages:
   ```
   pip install -r requirements.txt
   ```

### Step 5: Configure API Keys

1. Open the main Python file in a text editor
2. Find these lines:
   ```python
   GEMINI_API_KEY = "AIzaSyACFe5N9h_h7FgIPbPH-4GKkJMYhj712QQ"
   POLLINATIONS_API_KEY = "lLBgrZZfLZ-q6McF"
   ```
3. Replace them with your own API keys (see "Getting API Keys" section below)

## 🔑 Getting API Keys

### Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it in the script

### Pollinations API Key

1. Go to [Pollinations](https://pollinations.ai/)
2. Create an account or sign in
3. Navigate to your account settings
4. Find your API key and copy it
5. Paste it in the script

## 🎮 How to Use

1. Open Command Prompt as administrator
2. Navigate to the project folder
3. Activate the virtual environment:
   ```
   .venv\Scripts\activate
   ```
4. Run the application:
   ```
   python main.py
   ```
5. A debug window will appear in the bottom right corner
6. Type your command when prompted
7. Confirm you want to proceed
8. Watch the agent work its magic!

## 💡 Example Commands

- "Open Chrome browser and search for 'what are allotropes of carbon' then write it in notepad"
- "Open calculator and calculate 25 * 37"
- "Take a screenshot and save it to desktop"
- "Open notepad and write a short poem about computers"
- "Open File Explorer and navigate to Documents folder"
- "Open Spotify and play your liked songs"

## 🛠️ Understanding the Debug Window

The debug window shows you:
- What the agent is planning to do
- Each step as it's being executed
- Screenshots it's taking
- Any errors or issues
- Results of completed actions

You can minimize this window using the "Minimize" button, and it will automatically restore when needed.

## ⚠️ Safety Features

- **Emergency Stop**: Move your mouse to the top-left corner of the screen to immediately stop all actions
- **Confirmation Required**: The agent always asks for confirmation before executing tasks
- **Error Handling**: If something goes wrong, the agent will try to recover or stop safely
- **Failsafe**: Built-in protections prevent accidental harmful actions

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **"Calling Tcl from different apartment" Error**
   - This is a threading issue that has been fixed in the latest version
   - Make sure you're using the most recent code

2. **Debug Window Not Appearing**
   - Check if your antivirus is blocking the script
   - Try running the script with administrator privileges

3. **Tesseract Not Found Error**
   - Make sure you installed Tesseract correctly
   - Verify it was added to your system PATH
   - Try restarting your computer after installation

4. **API Key Errors**
   - Double-check that your API keys are correct
   - Make sure you have enough credits/quota for the services
   - Verify your internet connection is stable

5. **Coordinate Extraction Not Working**
   - The agent will fall back to alternative methods if needed
   - Make sure your screen resolution is set to a standard size
   - Check that the target application is fully visible

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Test everything thoroughly
5. Submit a pull request

For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Gemini 2.0 Flash for task planning
- Holo 1.5 for coordinate extraction
- Pollinations OpenAI for content processing
- The open-source community for the various libraries used

## ⚠️ Disclaimer

This tool automates interactions with your computer. While safety features are in place, use it at your own risk. The developers are not responsible for any unintended actions or data loss. Always review the planned steps before confirming execution.

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Search existing issues on GitHub
3. Create a new issue with details about your problem
4. Include screenshots of any error messages

---

**Happy automating!** 🎉