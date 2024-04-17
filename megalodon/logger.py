from datetime import timedelta
import sys
import time
import logging
import math


class LogFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def format(self, record):
        subsecond, seconds = math.modf(record.created)
        curr_date = (
            time.strftime("%y-%m-%d %H:%M:%S", time.localtime(seconds))
            + f".{int(subsecond * 1_000_000):06d}"
        )
        delta = timedelta(seconds=round(record.created - self.start_time))
        prefix = f"{record.levelname:<7} {curr_date} - {delta} - "

        content = record.getMessage()
        content = content.replace("\n", "\n" + " " * len(prefix))

        return f"{prefix}{content}"


def initialize_logger() -> logging.Logger:
    # log everything
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # stdout: everything
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.NOTSET)
    stdout_handler.setFormatter(LogFormatter())

    # stderr: warnings / errors and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(LogFormatter())

    # set stream handlers
    logger.handlers.clear()
    assert len(logger.handlers) == 0, logger.handlers
    logger.handlers.append(stdout_handler)
    logger.handlers.append(stderr_handler)

    return logger


def add_logger_file_handler(filepath: str):
    # build file handler
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(LogFormatter())

    # update logger
    logger = logging.getLogger()
    logger.addHandler(file_handler)
