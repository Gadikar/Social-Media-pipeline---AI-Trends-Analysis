import logging
from pyfaktory import Client, Producer, Job
import sys
import os
from dotenv import load_dotenv


logger = logging.getLogger("reddit cold start")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
logger.addHandler(sh)

if __name__ == "__main__":
    load_dotenv()
    subreddit_list_str = os.getenv("SUBREDDIT_LIST", "")
    subreddits = subreddit_list_str.split(",")
    print(f"Cold starting catalog crawl for subreddits: {', '.join(subreddits)}")
    faktory_server_url = os.environ.get("FAKTORY_SERVER_URL")

    with Client(faktory_url=faktory_server_url, role="producer") as client:
        producer = Producer(client=client)
        for subreddit in subreddits:
            job = Job(jobtype="crawl-subreddit", args=(subreddit,), queue="crawl-subreddit")
            producer.push(job)
            logger.info(f"Pushed job for subreddit: {subreddit}")
