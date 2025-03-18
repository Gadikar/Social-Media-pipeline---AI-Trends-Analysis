import logging
import os
from typing import List

from pyfaktory import Client, Consumer, Job, Producer
import time
import random
import sys
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger("faktory test")
logger.propagate = False
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
logger.addHandler(sh)

# Load environment variables
load_dotenv()

FAKTORY_SERVER_URL = os.environ.get("FAKTORY_SERVER_URL")


def get_boards_from_env() -> List[str]:
    boards_str = os.environ.get("4CHAN_BOARDS")
    if not boards_str:
        logger.error("No BOARDS found in environment variables")
        sys.exit(1)

    return [board.strip() for board in boards_str.split(",")]


if __name__ == "__main__":
    boards = get_boards_from_env()
    logger.info(f"Starting catalog crawl for boards: {', '.join(boards)}")

    with Client(faktory_url=FAKTORY_SERVER_URL, role="producer") as client:
        producer = Producer(client=client)

        for board in boards:
            logger.info(f"Queuing job for board: {board}")
            job = Job(jobtype="crawl-catalog", args=(board,), queue="crawl-catalog")
            producer.push(job)
            logger.info(f"Successfully queued job for board: {board}")