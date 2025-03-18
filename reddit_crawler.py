import json
from typing import Dict, Any

from cleaning import clean_text
from reddit_client import RedditClient
import logging
from pyfaktory import Client, Consumer, Job, Producer
import datetime
import psycopg2
from psycopg2.extras import Json
from psycopg2.extensions import register_adapter
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
from speechmoderate_client import ModerateHateSpeechClient

register_adapter(dict, Json)

logger = logging.getLogger("reddit crawler")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
logger.addHandler(sh)

load_dotenv()

FAKTORY_SERVER_URL = os.environ.get("FAKTORY_SERVER_URL")
DATABASE_URL = os.environ.get("DATABASE_URL")
MAX_POSTS_PER_CRAWL = 500  # Adjust based on your needs
MAX_COMMENTS_PER_POST = 1000  # Adjust based on your needs
def crawl_subreddit(subreddit):

    reddit_client = RedditClient()
    conn = psycopg2.connect(dsn=DATABASE_URL)
    cur = conn.cursor()
    hate_speech_client = ModerateHateSpeechClient()

    try:
        post_ids_array = []
        title_score = 0
        title_class = "none"
        for post in reddit_client.get_all_new_posts(subreddit, max_posts=MAX_POSTS_PER_CRAWL):
            post_data = post['data']
            post_id = post_data['id']
            if subreddit == 'politics':

            #process for hate  speech
                cleaned_title = clean_text(post_data.get('title', ''))
                cleaned_text = clean_text(post_data.get('selftext', ''))

                if cleaned_title is not None:
                    try:
                        response = hate_speech_client.check_comment(cleaned_title)
                        logger.warning(response)
                        result = process_response (response)
                        title_score = result["score"]
                        title_class = result["title_class"]
                        #logger.info(f"Score: {title_score} , class: {title_class}")
                    except Exception as e:
                        logger.info(f"Error in hate speech Api {e}")


                q = """     
                INSERT INTO reddit_posts (subreddit, post_id, data, last_comment_update, cleaned_title, cleaned_text ,title_score, title_class)
                VALUES (%s, %s, %s, NOW(), %s, %s, %s, %s)
                ON CONFLICT (post_id) DO UPDATE
                SET data = EXCLUDED.data, last_comment_update = NOW(), 
                cleaned_title = EXCLUDED.cleaned_title, cleaned_text = EXCLUDED.cleaned_text, title_score = EXCLUDED.title_score, title_class = EXCLUDED.title_class   
                RETURNING post_id
                """

                cur.execute(q, (subreddit, post_id, post_data, cleaned_title, cleaned_text, title_score, title_class))
                conn.commit()
            else:
                q = """
                         INSERT INTO reddit_posts (subreddit, post_id, data, last_comment_update)
                         VALUES (%s, %s, %s, NOW())
                         ON CONFLICT (post_id) DO UPDATE
                         SET data = EXCLUDED.data, last_comment_update = NOW()
                         RETURNING post_id
                         """
                cur.execute(q, (subreddit, post_id, post_data))

            db_id = cur.fetchone()
            if db_id:
                post_ids_array.append(post_id)

        with Client(faktory_url=FAKTORY_SERVER_URL, role="producer") as client:
            producer = Producer(client=client)
            job = Job(
                jobtype="crawl-comments",
                args=(subreddit, post_ids_array),
                queue="crawl-comments"
            )
            producer.push(job)
    
    finally:
        cur.close()
        conn.close()

    schedule_next_crawl(subreddit)
    schedule_update_old_posts(subreddit)


def process_reddit_comments(comment_data: dict, post_id: str) -> list[dict]:
    """
    Recursively process Reddit comments and their replies to create flattened records.
    """
    processed_comments = []
    
    try:
        # Process the base comment
        base_comment = {
            'id': comment_data['id'],
            'post_id': post_id,
            'parent_id': comment_data.get('parent_id', post_id),
            'data': comment_data,
            'comment_text': comment_data['body']
        }
        processed_comments.append(base_comment)
        
        # Process replies if they exist
        replies = comment_data.get('replies', '')
        if replies and isinstance(replies, dict):
            children = replies.get('data', {}).get('children', [])
            for child in children:
                child_data = child.get('data', {})
                processed_replies = process_reddit_comments(child_data, post_id)
                processed_comments.extend(processed_replies)
                
    except Exception as e:
        print(f"Error in process_reddit_comments: {e}")
        raise
        
    return processed_comments


def crawl_comments(subreddit, post_ids):
    reddit_client = RedditClient()
    conn = psycopg2.connect(dsn=DATABASE_URL)
    cur = conn.cursor()
    hate_speech_client = ModerateHateSpeechClient()

    try:
        # Process each post ID in the array
        for post_id in post_ids:
            cur.execute("SELECT last_comment_id FROM reddit_posts WHERE post_id = %s", (post_id,))
            result = cur.fetchone()
            last_comment_id = result[0] if result else None

            new_last_comment_id = last_comment_id
            for comment in reddit_client.get_all_post_comments(subreddit, post_id, max_comments=MAX_COMMENTS_PER_POST):
                comment_data = comment['data']
                comment_id = comment_data['id']

                # If we've reached the last processed comment, stop processing this post
                if comment_id == last_comment_id:
                    break

                # Update the new_last_comment_id if this is the first comment we're processing
                if new_last_comment_id == last_comment_id:
                    new_last_comment_id = comment_id

                # Process the comment and its replies
                processed_comments = process_reddit_comments(comment_data, post_id)
                
                # Insert into original reddit_comments table
                q_original = """
                INSERT INTO reddit_comments (subreddit, post_id, comment_id, data)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (comment_id) DO UPDATE
                SET data = EXCLUDED.data
                """
                cur.execute(q_original, (
                    subreddit,
                    post_id,
                    comment_id,
                    comment_data
                ))
                
                # Insert all processed comments into cleaned table
                for processed_comment in processed_comments:
                    cleaned_text= processed_comment['comment_text']
                    response = hate_speech_client.check_comment(cleaned_text)
                    logger.warning(response)
                    result = process_response (response)
                    title_score = result["score"]
                    title_class = result["title_class"]

                    q_cleaned = """
                    INSERT INTO reddit_comments_cleaned (id, post_id, parent_id, data, cleaned_text,score, class)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET post_id = EXCLUDED.post_id,
                        parent_id = EXCLUDED.parent_id,
                        data = EXCLUDED.data
                    """
                    cur.execute(q_cleaned, (
                        processed_comment['id'],
                        processed_comment['post_id'],
                        processed_comment['parent_id'],
                        processed_comment['data'],
                        processed_comment['comment_text'],
                        title_score,
                        title_class

                    ))

            # Update the last_comment_id for the current post
            if new_last_comment_id != last_comment_id:
                cur.execute(
                    "UPDATE reddit_posts SET last_comment_id = %s WHERE post_id = %s",
                    (new_last_comment_id, post_id)
                )

            # Commit after each post is processed
            conn.commit()

    finally:
        cur.close()
        conn.close()

def crawl_comment_single(subreddit, post_id):
    reddit_client = RedditClient()
    conn = psycopg2.connect(dsn=DATABASE_URL)
    cur = conn.cursor()
    hate_speech_client = ModerateHateSpeechClient()

    try:
        cur.execute("SELECT last_comment_id FROM reddit_posts WHERE post_id = %s", (post_id,))
        result = cur.fetchone()
        last_comment_id = result[0] if result else None

        new_last_comment_id = last_comment_id
        for comment in reddit_client.get_all_post_comments(subreddit, post_id, max_comments=MAX_COMMENTS_PER_POST):
            comment_data = comment['data']
            comment_id = comment_data['id']

            # If we've reached the last processed comment, stop processing
            if comment_id == last_comment_id:
                break

            # Update the new_last_comment_id if this is the first comment we're processing
            if new_last_comment_id == last_comment_id:
                new_last_comment_id = comment_id

            # Process the comment and its replies
            processed_comments = process_reddit_comments(comment_data, post_id)
            
            # Insert into original reddit_comments table
            q_original = """
            INSERT INTO reddit_comments (subreddit, post_id, comment_id, data)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (comment_id) DO UPDATE
            SET data = EXCLUDED.data
            """
            cur.execute(q_original, (
                subreddit,
                post_id,
                comment_id,
                comment_data
            ))
            
            # Insert all processed comments into cleaned table
            for processed_comment in processed_comments:
                cleaned_text= processed_comment['comment_text']
                response = hate_speech_client.check_comment(cleaned_text)
                logger.warning(response)
                result = process_response (response)
                title_score = result["score"]
                title_class = result["title_class"]

                q_cleaned = """
                INSERT INTO reddit_comments_cleaned (id, post_id, parent_id, data, cleaned_text,score, class)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET post_id = EXCLUDED.post_id,
                    parent_id = EXCLUDED.parent_id,
                    data = EXCLUDED.data
                """
                cur.execute(q_cleaned, (
                    processed_comment['id'],
                    processed_comment['post_id'],
                    processed_comment['parent_id'],
                    processed_comment['data'],
                    processed_comment['comment_text'],
                    title_score,
                    title_class
                ))

        # Update the last_comment_id for the post
        if new_last_comment_id != last_comment_id:
            cur.execute(
                "UPDATE reddit_posts SET last_comment_id = %s WHERE post_id = %s",
                (new_last_comment_id, post_id)
            )

        conn.commit()

    finally:
        cur.close()
        conn.close()

        
def update_old_posts(subreddit):
    conn = psycopg2.connect(dsn=DATABASE_URL)
    cur = conn.cursor()

    # Get posts from the last 3 days that haven't been updated in the last 6 hours
    three_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=7)
    six_hours_ago = datetime.datetime.utcnow() - datetime.timedelta(hours=6)
    
    q = """
    SELECT post_id FROM reddit_posts 
    WHERE subreddit = %s 
    AND (data->>'created_utc')::float > %s 
    AND (last_comment_update IS NULL OR last_comment_update < %s)
    ORDER BY (data->>'created_utc')::float DESC 
    """
    
    cur.execute(q, (subreddit, three_days_ago.timestamp(), six_hours_ago))
    
    old_posts = cur.fetchall()

    cur.close()
    conn.close()
    logger.info(f"########## OLD POST PRINT ############## \n {old_posts}")
    with Client(faktory_url=FAKTORY_SERVER_URL, role="producer") as client:
        producer = Producer(client=client)
        job = Job(
            jobtype="crawl-comments",
            args=(subreddit, old_posts),
            queue="crawl-comments"
        )
        producer.push(job)
        

def schedule_next_crawl(subreddit):
    with Client(faktory_url=FAKTORY_SERVER_URL, role="producer") as client:
        producer = Producer(client=client)
        run_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
        run_at = run_at.isoformat()[:-7] + "Z"
        logger.info(f"Next crawl for {subreddit} scheduled at: {run_at}")
        job = Job(
            jobtype="crawl-subreddit",
            args=(subreddit,),
            queue="crawl-subreddit",
            at=str(run_at),
        )
        producer.push(job)

def schedule_update_old_posts(subreddit):
    with Client(faktory_url=FAKTORY_SERVER_URL, role="producer") as client:
        producer = Producer(client=client)
        run_at = datetime.datetime.utcnow() + datetime.timedelta(hours=6)  # Run every 2 hours
        run_at = run_at.isoformat()[:-7] + "Z"
        logger.info(f"Next old post update for {subreddit} scheduled at: {run_at}")
        job = Job(
            jobtype="update-old-posts",
            args=(subreddit,),
            queue="update-old-posts",
            at=str(run_at),
        )
        producer.push(job)

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


if __name__ == "__main__":
    with Client(faktory_url=FAKTORY_SERVER_URL, role="consumer") as client:
        consumer = Consumer(
            client=client, 
            queues=["crawl-subreddit", "crawl-comments", "update-old-posts","crawl-comment-single"], 
            concurrency=1
        )
        consumer.register("crawl-subreddit", crawl_subreddit)
        consumer.register("crawl-comments", crawl_comments)
        consumer.register("update-old-posts", update_old_posts)
        consumer.register("crawl-comment-single", crawl_comment_single)
        consumer.run()
        
