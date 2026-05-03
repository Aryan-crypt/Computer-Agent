import logging

# ==========================================
# LOG NOISE SUPPRESSION
# Silences overly verbose third-party libraries 
# added for the new features (Scheduler, Reconnect, Vision)
# ==========================================
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.getLogger("apscheduler.executors.default").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.ERROR) # Silences PTB internal heartbeat logs

from OmniCtrl_Agent.telegram_interface import TelegramPCInterface

def main():
    """Entry point for the Advanced PC Control Agent"""
    
    # Set up clean, readable logging for OUR application only
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S' # Cleaner timestamp format
    )
    
    print("Initializing PC Control Agent...")
    bot = TelegramPCInterface()
    
    # Starts the infinite loop with autonomous self-reconnection
    bot.run()

if __name__ == "__main__":
    main()