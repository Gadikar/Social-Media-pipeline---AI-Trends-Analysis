import requests
import logging
import time
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import os
from typing import List, Dict, Optional, Generator, Any

load_dotenv()

logger = logging.getLogger("reddit client")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
logger.addHandler(sh)

class RedditClient:
    BASE_URL = 'https://oauth.reddit.com/r/{subreddit}/new'
    COMMENT_URL = 'https://oauth.reddit.com/r/{subreddit}/comments/{article_id}'
    TOKEN_URL = 'https://www.reddit.com/api/v1/access_token'

    def __init__(self):
        load_dotenv()
        self.client_id = os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.username = os.getenv('REDDIT_USERNAME')
        self.password = os.getenv('REDDIT_PASSWORD')
        self.user_agent = 'app/1.0'
        self.access_token = None
        self.token_expires_at = 0
        self.session = requests.Session()

    def _get_access_token(self):
        auth = HTTPBasicAuth(self.client_id, self.client_secret)
        data = {
            'grant_type': 'password',
            'username': self.username,
            'password': self.password,
        }
        headers = {'User-Agent': self.user_agent}
        response = self.session.post(self.TOKEN_URL, auth=auth, data=data, headers=headers)
        response.raise_for_status()
        token_data = response.json()
        self.access_token = token_data['access_token']
        self.token_expires_at = time.time() + token_data['expires_in']

    def _ensure_valid_token(self):
        if time.time() >= self.token_expires_at:
            self._get_access_token()

    def get_all_new_posts(self, subreddit: str, max_posts: Optional[int] = None) -> Generator[Dict, None, None]:
        """
        Generator that yields all new posts from a subreddit using pagination.
        
        Args:
            subreddit: The subreddit to fetch posts from
            max_posts: Maximum number of posts to fetch (None for all available)
        """
        count = 0
        after = None
        posts_fetched = 0
        
        while True:
            batch = self._get_new_posts_page(subreddit, after=after, count=count)
            if not batch or not batch['data']['children']:
                break
                
            for post in batch['data']['children']:
                yield post
                posts_fetched += 1
                if max_posts and posts_fetched >= max_posts:
                    return

            after = batch['data']['after']
            if not after:
                break
                
            count += len(batch['data']['children'])
            # Reddit's rate limiting - sleep to avoid hitting limits
            time.sleep(2)

    def _get_new_posts_page(self, subreddit: str, after: Optional[str] = None, 
                           count: int = 0, limit: int = 100) -> Dict:
        """
        Fetch a single page of new posts from a subreddit.
        """
        self._ensure_valid_token()
        url = self.BASE_URL.format(subreddit=subreddit)
        params = {
            'limit': limit,
            'count': count
        }
        if after:
            params['after'] = after
            
        return self._make_request(url, params)

    def get_all_post_comments(self, subreddit: str, article_id: str, 
                            max_comments: Optional[int] = None) -> Generator[Dict, None, None]:
        """
        Generator that yields all comments from a post using pagination.
        
        Args:
            subreddit: The subreddit containing the post
            article_id: The post ID to fetch comments from
            max_comments: Maximum number of comments to fetch (None for all available)
        """
        comments_fetched = 0
        more_comments = set()
        initial_comments = self._get_post_comments_page(subreddit, article_id)
        
        # Process top-level comments first
        for comment in initial_comments[1]['data']['children']:
            if comment['kind'] == 't1':  # Regular comment
                yield comment
                comments_fetched += 1
                if max_comments and comments_fetched >= max_comments:
                    return
            elif comment['kind'] == 'more':  # More comments to fetch
                more_comments.update(comment['data']['children'])

        # Fetch additional comments in batches
        while more_comments and (not max_comments or comments_fetched < max_comments):
            batch_size = min(100, len(more_comments))  # Reddit's API limit
            comment_batch = list(more_comments)[:batch_size]
            more_comments = more_comments - set(comment_batch)
            
            additional_comments = self._get_more_comments(subreddit, article_id, comment_batch)
            for comment in additional_comments:
                if comment['kind'] == 't1':
                    yield comment
                    comments_fetched += 1
                    if max_comments and comments_fetched >= max_comments:
                        return
                elif comment['kind'] == 'more':
                    more_comments.update(comment['data']['children'])
            
            time.sleep(2)  # Rate limiting

    def _get_post_comments_page(self, subreddit: str, article_id: str) -> List[Dict]:
        """
        Fetch a single page of comments from a post.
        """
        self._ensure_valid_token()
        url = self.COMMENT_URL.format(subreddit=subreddit, article_id=article_id)
        return self._make_request(url)

    def _get_more_comments(self, subreddit: str, article_id: str, comment_ids: List[str]) -> List[Dict]:
        """
        Fetch additional comments using the /api/morechildren endpoint.
        """
        self._ensure_valid_token()
        url = 'https://oauth.reddit.com/api/morechildren'
        params = {
            'api_type': 'json',
            'link_id': f't3_{article_id}',
            'children': ','.join(comment_ids)
        }
        response = self._make_request(url, params)
        return response.get('json', {}).get('data', {}).get('things', [])

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Any:
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'User-Agent': self.user_agent
        }
        while True:
            try:
                response = self.session.get(url, headers=headers, params=params)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting for {retry_after} seconds.")
                    time.sleep(retry_after)
                else:
                    logger.error(f"HTTP error occurred: {e}")
                    raise
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                raise

if __name__ == "__main__":
    client = RedditClient()
    # Example usage
    for i, post in enumerate(client.get_all_new_posts('artificial', max_posts=10)):
        print(f"Post {i + 1}: {post['data']['title']}")
        for j, comment in enumerate(client.get_all_post_comments('artificial', post['data']['id'], max_posts=5)):
            print(f"  Comment {j + 1}: {comment['data']['body'][:100]}...")