CREATE TABLE IF NOT EXISTS public.chan_posts
(
    id BIGSERIAL NOT NULL,
    post_number bigint NOT NULL,
    data jsonb NOT NULL,
    thread_number bigint NOT NULL,
    board text COLLATE pg_catalog."default" NOT NULL,
    CONSTRAINT posts_pkey PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS posts_board_thread_number_post_number_idx
    ON public.chan_posts USING btree
    (board COLLATE pg_catalog."default" ASC NULLS LAST, thread_number ASC NULLS LAST, post_number ASC NULLS LAST);

CREATE UNIQUE INDEX IF NOT EXISTS posts_board_thread_number_post_number_idx1
    ON public.chan_posts USING btree
    (board COLLATE pg_catalog."default" ASC NULLS LAST, thread_number ASC NULLS LAST, post_number ASC NULLS LAST);

CREATE INDEX IF NOT EXISTS posts_post_number_idx
    ON public.chan_posts USING btree
    (post_number ASC NULLS LAST);

CREATE INDEX IF NOT EXISTS posts_thread_number_post_number_idx
    ON public.chan_posts USING btree
    (thread_number ASC NULLS LAST, post_number ASC NULLS LAST);
