CREATE TABLE IF NOT EXISTS public.reddit_posts
(
    post_id text COLLATE pg_catalog."default" NOT NULL,
    subreddit text COLLATE pg_catalog."default" NOT NULL,
    data jsonb NOT NULL,
    last_comment_id text COLLATE pg_catalog."default",
    last_comment_update timestamp without time zone,
    CONSTRAINT reddit_posts_pkey PRIMARY KEY (post_id)
);

CREATE INDEX IF NOT EXISTS idx_reddit_posts_last_comment_update
    ON public.reddit_posts USING btree
    (last_comment_update ASC NULLS LAST);

CREATE INDEX IF NOT EXISTS idx_reddit_posts_subreddit
    ON public.reddit_posts USING btree
    (subreddit COLLATE pg_catalog."default" ASC NULLS LAST);

CREATE TABLE IF NOT EXISTS public.reddit_comments
(
    subreddit text COLLATE pg_catalog."default" NOT NULL,
    post_id text COLLATE pg_catalog."default" NOT NULL,
    comment_id text COLLATE pg_catalog."default" NOT NULL,
    data jsonb NOT NULL,
    is_deleted boolean DEFAULT false,
    CONSTRAINT reddit_comments_pkey PRIMARY KEY (comment_id),
    CONSTRAINT post_id FOREIGN KEY (post_id)
        REFERENCES public.reddit_posts (post_id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
        NOT VALID
);

CREATE INDEX IF NOT EXISTS idx_reddit_comments_is_deleted
    ON public.reddit_comments USING btree
    (is_deleted ASC NULLS LAST)
    WHERE is_deleted = false;

CREATE INDEX IF NOT EXISTS idx_reddit_comments_subreddit
    ON public.reddit_comments USING btree
    (subreddit COLLATE pg_catalog."default" ASC NULLS LAST);