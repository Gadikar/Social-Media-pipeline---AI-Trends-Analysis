from chan_client import ChanClient
import logging
from pyfaktory import Client, Consumer, Job, Producer
import datetime
import psycopg2
from speechmoderate_client import ModerateHateSpeechClient
from cleaning import clean_text
from typing import Dict, Any
import json

# these three lines allow psycopg to insert a dict into
# a jsonb coloumn
from psycopg2.extras import Json
from psycopg2.extensions import register_adapter

register_adapter(dict, Json)

# load in function for .env reading
from dotenv import load_dotenv


logger = logging.getLogger("4chan client")
logger.propagate = False
logger.setLevel(logging.INFO)

sh = logging.StreamHandler()

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s"
)

sh.setFormatter(formatter)
logger.addHandler(sh)

load_dotenv()

import os

FAKTORY_SERVER_URL = os.environ.get("FAKTORY_SERVER_URL")
DATABASE_URL = os.environ.get("DATABASE_URL")

hate_speech_client = ModerateHateSpeechClient()

"""
Return all the thread numbers from a catalog json object
"""


def thread_numbers_from_catalog(catalog):
    thread_numbers = []
    for page in catalog:
        for thread in page["threads"]:
            thread_number = thread["no"]
            thread_numbers.append(thread_number)

    return thread_numbers


"""
Return thread numbers that existed in previous but don't exist
in current
"""


def find_dead_threads(previous_catalog_thread_numbers, current_catalog_thread_numbers):
    dead_thread_numbers = set(previous_catalog_thread_numbers).difference(
        set(current_catalog_thread_numbers)
    )
    return dead_thread_numbers


def process_response(response_data: str) -> Dict[str, Any]:
    try:
        # First try to parse the JSON if response is a string
        if isinstance(response_data, str):
            response = json.loads(response_data)
        else:
            response = response_data

        logger.warning(f"Received response: {response}")

        if isinstance(response, dict):
            try:
                score = float(response.get('confidence', 0.0))
                title_class = str(response.get('class', ''))
                title_score = f"{score:.2f}"
                return {
                    'score': score,
                    'title_class': title_class,
                    'title_score': title_score
                }
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing values in response: {e}")
                return {
                    'score': 0.0,
                    'title_class': '',
                    'title_score': '0.00'
                }
        else:
            logger.error("Response is not a dictionary")
            return {
                'score': 0.0,
                'title_class': '',
                'title_score': '0.00'
            }

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON response: {e}")
        return {
            'score': 0.0,
            'title_class': '',
            'title_score': '0.00'
        }
    except Exception as e:
        logger.error(f"Unexpected error processing response: {e}")
        return {
            'score': 0.0,
            'title_class': '',
            'title_score': '0.00'
        }
"""
Crawl a given thread and get its json.
Insert the posts into db
"""


def crawl_thread(board, thread_number):
    chan_client = ChanClient()
    thread_data = chan_client.get_thread(board, thread_number)

    logger.info(f"Thread: {board}/{thread_number}/:\n{thread_data}")

    # really soould use a connection pool
    conn = psycopg2.connect(dsn=DATABASE_URL)

    cur = conn.cursor()
    # now insert into db
    # iterate through the thread data and get all the post data
    for post in thread_data["posts"]:
        post_number = post["no"]
        created_at = post["time"]
        # Handle cases where 'com' field might not exist
        if board == 'pol':
            cleaned_text = clean_text(post.get("com", "")) if "com" in post else ""

            response = hate_speech_client.check_comment(cleaned_text)
            logger.warning(response)
            result = process_response(response_data=response)

            title_score = result["score"]
            title_class = result["title_class"]

            # Fixed SQL query - removed extra placeholder
            q = """
                                   INSERT INTO chan_posts 
                                   (board, thread_number, post_number, data,created_at, score, class, cleaned_text) 
                                   VALUES (%s, %s, %s, %s, %s, %s, %s ,%s) 
                                   RETURNING id
                               """
            cur.execute(q, (
                board,
                thread_number,
                post_number,
                json.dumps(post),  # Ensure post data is JSON serialized'
                created_at,
                title_score,
                title_class,
                cleaned_text
            ))
        else:
            cleaned_text = clean_text(post.get("com", "")) if "com" in post else ""

            q = "INSERT INTO chan_posts (board, thread_number, post_number, data,created_at, cleaned_text) VALUES (%s, %s, %s, %s, %s) RETURNING id"
            cur.execute(q, (board, thread_number, post_number,post,created_at,cleaned_text))
        conn.commit()

        db_id = cur.fetchone()[0]
        logger.info(f"Inserted post {post_number} with DB id: {db_id}")

    # close cursor connection
    cur.close()
    # close connection
    conn.close()


"""
Go out, grab the catalog for a given board, and figure out what threads we need
to collect.

For each thread to collect, enqueue a new job to crawl the thread.

Schedule catalog crawl to run again at some point in the future.
"""


def crawl_catalog(board, previous_catalog_thread_numbers=[]):
    chan_client = ChanClient()

    current_catalog = chan_client.get_catalog(board)

    current_catalog_thread_numbers = thread_numbers_from_catalog(current_catalog)

    dead_threads = find_dead_threads(
        previous_catalog_thread_numbers, current_catalog_thread_numbers
    )
    logger.info(f"dead threads: {dead_threads}")

    # issue the crawl thread jobs for each dead thread
    crawl_thread_jobs = []
    with Client(faktory_url=FAKTORY_SERVER_URL, role="producer") as client:
        producer = Producer(client=client)
        for dead_thread in dead_threads:
            # see https://github.com/ghilesmeddour/faktory_worker_python/blob/main/src/pyfaktory/models.py
            # what a `Job` looks like
            job = Job(
                jobtype="crawl-thread", args=(board, dead_thread), queue="crawl-thread"
            )

            crawl_thread_jobs.append(job)

        producer.push_bulk(crawl_thread_jobs)

    # Schedule another catalog crawl to happen at some point in future
    with Client(faktory_url=FAKTORY_SERVER_URL, role="producer") as client:
        producer = Producer(client=client)
        # figure out how to use non depcreated methods on your own
        # run_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)
        run_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
        run_at = run_at.isoformat()[:-7] + "Z"
        logger.info(f"run_at = {run_at}")
        job = Job(
            jobtype="crawl-catalog",
            args=(board, current_catalog_thread_numbers),
            queue="crawl-catalog",
            at=str(run_at),
        )
        producer.push(job)


if __name__ == "__main__":
    # we want to pull jobs off the queues and execute them
    # FOREVER (continuously)
    with Client(faktory_url=FAKTORY_SERVER_URL, role="consumer") as client:
        consumer = Consumer(
            client=client, queues=["crawl-catalog", "crawl-thread"], concurrency=5
        )
        consumer.register("crawl-catalog", crawl_catalog)
        consumer.register("crawl-thread", crawl_thread)
        # tell the consumer to pull jobs off queue and execute them!
        consumer.run()
