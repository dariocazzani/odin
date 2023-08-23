import logging
import coloredlogs

class ColoredLogger:
    def __init__(self, logger_name: str, level: str = 'INFO'):
        """
        Initialize a logger with a given name and level.
        
        Args:
            logger_name (str): Name of the logger.
            level (str): Logging level. Default is 'INFO'.
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)
        
        coloredlogs.install(level=level, logger=self.logger)

    def get_logger(self):
        """
        Returns the initialized logger.
        """
        return self.logger

# Sample usage:
if __name__ == "__main__":
    log = ColoredLogger('test_logger').get_logger()
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")
