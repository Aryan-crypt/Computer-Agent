from OmniCtrl_Agent.telegram_interface import *

def main():
    """Entry point"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    bot = TelegramPCInterface()
    bot.run()


if __name__ == "__main__":
    main()