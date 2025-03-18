from fastapi import FastAPI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace '*' with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
DB_URL = 'postgresql://postgres:password@localhost:5433/crawler_db'
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@app.get("/chan/stats")
async def get_detailed_stats():
    with engine.connect() as conn:
        query = text("""
           WITH board_stats AS (
    SELECT
        board,
        COUNT(*) AS total_posts,
        AVG((data->>'replies')::INT) AS average_replies,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (data->>'replies')::INT) AS median_replies,
        AVG((data->>'images')::INT) AS average_images_per_post,
        ROUND(SUM(CASE WHEN (data->>'images')::INT > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS percentage_with_images,
        AVG((data->>'fsize')::INT) AS average_file_size,
        SUM(CASE WHEN (data->>'archived')::INT = 1 THEN 1 ELSE 0 END) AS archived_posts,
        ROUND(SUM(CASE WHEN (data->>'archived')::INT = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS percentage_archived,
        SUM(CASE WHEN (data->>'closed')::INT = 1 THEN 1 ELSE 0 END) AS closed_posts,
        ROUND(SUM(CASE WHEN (data->>'closed')::INT = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS percentage_closed,
        COUNT(DISTINCT thread_number) AS total_threads,
        ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT thread_number), 2) AS average_posts_per_thread
    FROM chan_posts
    GROUP BY board
)
SELECT
    'All Boards' AS board,  -- Combine results for all boards
    SUM(total_posts) AS total_posts,
    AVG(average_replies) AS average_replies,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY median_replies) AS median_replies,
    AVG(average_images_per_post) AS average_images_per_post,
    AVG(percentage_with_images) AS percentage_with_images,
    AVG(average_file_size) AS average_file_size,
    SUM(archived_posts) AS archived_posts,
    AVG(percentage_archived) AS percentage_archived,
    SUM(closed_posts) AS closed_posts,
    AVG(percentage_closed) AS percentage_closed,
    SUM(total_threads) AS total_threads,
    AVG(average_posts_per_thread) AS average_posts_per_thread
FROM board_stats;

        """)
        result = conn.execute(query)
        stats = [dict(zip(result.keys(), row)) for row in result.fetchall()]
        return stats

@app.get("/reddit/stats")
async def get_detailed_stats():
    with engine.connect() as conn:
        query = text("""
           SELECT 
               COUNT(DISTINCT rp.post_id) AS total_posts,
               COUNT(rc.comment_id) AS total_comments,
               ROUND(AVG(CASE WHEN rc.comment_id IS NOT NULL THEN 1 ELSE 0 END), 2) AS avg_comments_per_post,
               ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY JSONB_ARRAY_LENGTH(rc.data->'replies'::jsonb)), 2) AS median_replies_per_comment,
               COUNT(CASE WHEN rc.is_deleted = TRUE THEN 1 END) AS deleted_comments,
               ROUND(COUNT(CASE WHEN rc.is_deleted = TRUE THEN 1 END) * 100.0 / COUNT(rc.comment_id), 2) AS percent_deleted_comments,
               ROUND(SUM((rp.data->>'score')::INT) / COUNT(rp.post_id), 2) AS avg_post_score,
               ROUND(SUM((rc.data->>'score')::INT) / NULLIF(COUNT(rc.comment_id), 0), 2) AS avg_comment_score
           FROM reddit_posts rp
           LEFT JOIN reddit_comments rc ON rp.post_id = rc.post_id;
        """)
        result = conn.execute(query)
        stats = [dict(zip(result.keys(), row)) for row in result.fetchall()]
        return stats


