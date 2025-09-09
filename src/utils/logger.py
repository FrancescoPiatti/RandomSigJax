import logging

class Logger:
    def __init__(self, filepath: str, level: int = logging.INFO):
        self.logger = logging.getLogger(filepath)   # use filepath as unique name
        self.logger.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(filepath, mode="a")
        file_handler.setFormatter(formatter)

        # avoid duplicate handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

    def log(self, msg: str, level: int = logging.INFO):
        """
        Log a message at a given level
        """
        self.logger.log(level, msg)