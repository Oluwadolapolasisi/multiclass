"""
train.py

tensorboard


"""

import tensorflow as tf
from logger_setup import setup_logger, logger

#DATA_DIR = "data/train.csv"
 
def main():
    setup_logger()
    logger.info("Running")


if __name__ == "__main__":
    main()
