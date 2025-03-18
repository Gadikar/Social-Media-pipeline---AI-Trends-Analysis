import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from collections import defaultdict
from datetime import datetime, timedelta
import os

from community import community_louvain
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import hashlib
import pickle
from pathlib import Path
import community

# Configure cache directory
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Cache configuration
CACHE_TTL = 3600  # 1 hour in seconds


def get_cache_key(prefix, *args):
    """Generate a cache key based on arguments"""
    key = f"{prefix}:" + ":".join(str(arg) for arg in args)
    return hashlib.md5(key.encode()).hexdigest()


def save_cache(key, data):
    """Save data to cache file"""
    cache_file = CACHE_DIR / f"{key}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump({'timestamp': datetime.now(), 'data': data}, f)


def load_cache(key):
    """Load data from cache file if it exists and is not expired"""
    cache_file = CACHE_DIR / f"{key}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            if datetime.now() - cache_data['timestamp'] < timedelta(seconds=CACHE_TTL):
                return cache_data['data']
    return None


@st.cache_resource
def init_connection():
    """Initialize database connection"""
    db_params = {
        'dbname': os.getenv("DB_NAME"),
        'user': os.getenv("DB_USER"),
        'password': os.getenv("DB_PASSWORD"),
        'host': os.getenv("DB_HOST"),
        'port': os.getenv("DB_PORT", "5432")
    }

    connection_string = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
    return create_engine(connection_string)


@st.cache_data(ttl=3600)
def get_date_range(platform):
    """Get min and max dates from the database"""
    engine = init_connection()
    cache_key = get_cache_key('date_range', platform)

    cached_data = load_cache(cache_key)
    if cached_data is not None:
        return cached_data

    if platform == 'reddit':
        query = """
        SELECT 
            MIN(CAST(FLOOR(CAST(data->>'created' AS float)) AS bigint)) as min_date,
            MAX(CAST(FLOOR(CAST(data->>'created' AS float)) AS bigint)) as max_date
        FROM reddit_posts
        """
    else:  # 4chan
        query = """
        SELECT 
            MIN(CAST(FLOOR(CAST(data->>'time' AS float)) AS bigint)) as min_date,
            MAX(CAST(FLOOR(CAST(data->>'time' AS float)) AS bigint)) as max_date
        FROM chan_posts
        """

    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
        result = (df.iloc[0]['min_date'], df.iloc[0]['max_date'])
        save_cache(cache_key, result)
        return result



@st.cache_data(ttl=3600)
def load_reddit_data_chunked(start_date, end_date):
    """Load Reddit data with caching"""
    engine = init_connection()
    cache_key = get_cache_key('reddit_data', start_date, end_date)

    cached_data = load_cache(cache_key)
    if cached_data is not None:
        return cached_data

    query = f"""
    WITH hourly_stats AS (
        SELECT 
            DATE_TRUNC('hour', 
                      TO_TIMESTAMP(CAST(FLOOR(CAST(p.data->>'created' AS float)) AS bigint))
                     ) AT TIME ZONE 'UTC' AS hour,
            COUNT(DISTINCT p.post_id) as post_count,
            COUNT(c.id) as comment_count,
            COALESCE(AVG(c.score), 0) as avg_comment_score,
            p.subreddit
        FROM reddit_posts p
        LEFT JOIN reddit_comments_cleaned c ON p.post_id = c.post_id
        WHERE CAST(FLOOR(CAST(p.data->>'created' AS float)) AS bigint) 
            BETWEEN {start_date} AND {end_date}
        GROUP BY DATE_TRUNC('hour', 
                 TO_TIMESTAMP(CAST(FLOOR(CAST(p.data->>'created' AS float)) AS bigint))), 
                 p.subreddit
    )
    SELECT 
        hour,
        subreddit,
        post_count,
        comment_count,
        avg_comment_score
    FROM hourly_stats
    ORDER BY hour
    """

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
            if df.empty:
                st.warning("No data found for the selected date range.")
                return pd.DataFrame(columns=['hour', 'subreddit', 'post_count', 'comment_count', 'avg_comment_score'])
            save_cache(cache_key, df)
            return df
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame(columns=['hour', 'subreddit', 'post_count', 'comment_count', 'avg_comment_score'])


@st.cache_data(ttl=3600)
def load_4chan_data_chunked(start_date, end_date):
    """Load 4chan data with caching"""
    engine = init_connection()
    cache_key = get_cache_key('4chan_data', start_date, end_date)

    cached_data = load_cache(cache_key)
    if cached_data is not None:
        return cached_data

    query = f"""
    WITH hourly_stats AS (
        SELECT 
            DATE_TRUNC('hour', 
                      TO_TIMESTAMP(CAST(FLOOR(CAST(data->>'time' AS float)) AS bigint))
                     ) AT TIME ZONE 'UTC' AS hour,
            board,
            COUNT(*) as post_count,
            AVG(CAST(data->>'replies' AS INTEGER)) as avg_replies
        FROM chan_posts
        WHERE CAST(FLOOR(CAST(data->>'time' AS float)) AS bigint)
            BETWEEN {start_date} AND {end_date}
        GROUP BY DATE_TRUNC('hour', 
                 TO_TIMESTAMP(CAST(FLOOR(CAST(data->>'time' AS float)) AS bigint))),
                 board
    )
    SELECT 
        hour,
        board,
        post_count,
        avg_replies
    FROM hourly_stats
    ORDER BY hour
    """

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(query, conn)
            if df.empty:
                st.warning("No data found for the selected date range.")
                return pd.DataFrame(columns=['hour', 'board', 'post_count', 'avg_replies'])
            save_cache(cache_key, df)
            return df
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame(columns=['hour', 'board', 'post_count', 'avg_replies'])


@st.cache_data(ttl=3600)
def analyze_hourly_patterns(df, platform):
    """Analyze hourly patterns with caching"""
    try:
        if df.empty:
            if platform == 'Reddit':
                return pd.DataFrame(columns=['hour', 'subreddit', 'post_count', 'comment_count', 'avg_comment_score'])
            else:
                return pd.DataFrame(columns=['hour', 'board', 'post_count', 'avg_replies'])

        df_copy = df.copy()
        df_copy['hour'] = pd.to_datetime(df_copy['hour'], utc=True)
        df_copy['hour_num'] = df_copy['hour'].dt.hour

        # Group by hour number only, averaging across all days
        if platform == 'Reddit':
            hourly_stats = df_copy.groupby('hour_num').agg({
                'post_count': 'mean',
                'comment_count': 'mean',
                'avg_comment_score': 'mean'
            }).reset_index()
            hourly_stats = hourly_stats.rename(columns={'hour_num': 'hour'})
        else:
            hourly_stats = df_copy.groupby('hour_num').agg({
                'post_count': 'mean',
                'avg_replies': 'mean'
            }).reset_index()
            hourly_stats = hourly_stats.rename(columns={'hour_num': 'hour'})

        # Sort by hour to ensure proper line connection
        return hourly_stats.sort_values('hour')
    except Exception as e:
        st.error(f"Error in analyze_hourly_patterns: {str(e)}")
        if platform == 'Reddit':
            return pd.DataFrame(columns=['hour', 'subreddit', 'post_count', 'comment_count', 'avg_comment_score'])
        else:
            return pd.DataFrame(columns=['hour', 'board', 'post_count', 'avg_replies'])


@st.cache_data(ttl=3600)
def analyze_daily_patterns(df, platform):
    """Analyze daily patterns with caching"""
    try:
        if df.empty:
            if platform == 'Reddit':
                return pd.DataFrame(columns=['day', 'subreddit', 'post_count', 'comment_count', 'avg_comment_score'])
            else:
                return pd.DataFrame(columns=['day', 'board', 'post_count', 'avg_replies'])

        df_copy = df.copy()
        df_copy['hour'] = pd.to_datetime(df_copy['hour'], utc=True)

        if platform == 'Reddit':
            df_copy['day'] = df_copy['hour'].dt.day_name()
            daily_stats = df_copy.groupby(['day']).agg({
                'post_count': 'mean',
                'comment_count': 'mean',
                'avg_comment_score': 'mean'
            }).reset_index()
        else:
            df_copy['day'] = df_copy['hour'].dt.day_name()
            daily_stats = df_copy.groupby(['day']).agg({
                'post_count': 'mean',
                'avg_replies': 'mean'
            }).reset_index()

        # Sort days in correct order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_stats['day'] = pd.Categorical(daily_stats['day'], categories=day_order, ordered=True)
        return daily_stats.sort_values('day')
    except Exception as e:
        st.error(f"Error in analyze_daily_patterns: {str(e)}")
        if platform == 'Reddit':
            return pd.DataFrame(columns=['day', 'subreddit', 'post_count', 'comment_count', 'avg_comment_score'])
        else:
            return pd.DataFrame(columns=['day', 'board', 'post_count', 'avg_replies'])


def display_comprehensive_engagement_summary(hourly_stats, daily_stats, platform):
    """Display combined summary of hourly and daily engagement patterns"""
    st.subheader("📊 Comprehensive Engagement Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⏰ Hourly Patterns")
        if platform == "Reddit":
            peak_hour_posts = hourly_stats.loc[hourly_stats['post_count'].idxmax()]
            peak_hour_comments = hourly_stats.loc[hourly_stats['comment_count'].idxmax()]

            metrics = {
                "Peak Posting Hour": f"{int(peak_hour_posts['hour']):02d}:00 UTC",
                "Average Posts per Hour": f"{hourly_stats['post_count'].mean():.1f}",
                "Peak Comments Hour": f"{int(peak_hour_comments['hour']):02d}:00 UTC",
                "Average Comments per Hour": f"{hourly_stats['comment_count'].mean():.1f}"
            }
        else:
            peak_hour_posts = hourly_stats.loc[hourly_stats['post_count'].idxmax()]
            peak_hour_replies = hourly_stats.loc[hourly_stats['avg_replies'].idxmax()]

            metrics = {
                "Peak Posting Hour": f"{int(peak_hour_posts['hour']):02d}:00 UTC",
                "Average Posts per Hour": f"{hourly_stats['post_count'].mean():.1f}",
                "Peak Reply Hour": f"{int(peak_hour_replies['hour']):02d}:00 UTC",
                "Average Replies per Post": f"{hourly_stats['avg_replies'].mean():.2f}"
            }

        for key, value in metrics.items():
            st.metric(key, value)

    with col2:
        st.markdown("### 📅 Daily Patterns")
        if platform == "Reddit":
            peak_day_posts = daily_stats.loc[daily_stats['post_count'].idxmax()]
            peak_day_comments = daily_stats.loc[daily_stats['comment_count'].idxmax()]

            metrics = {
                "Most Active Day (Posts)": peak_day_posts['day'],
                "Average Daily Posts": f"{daily_stats['post_count'].mean():.1f}",
                "Most Active Day (Comments)": peak_day_comments['day'],
                "Average Daily Comments": f"{daily_stats['comment_count'].mean():.1f}"
            }
        else:
            peak_day_posts = daily_stats.loc[daily_stats['post_count'].idxmax()]
            peak_day_replies = daily_stats.loc[daily_stats['avg_replies'].idxmax()]

            metrics = {
                "Most Active Day (Posts)": peak_day_posts['day'],
                "Average Daily Posts": f"{daily_stats['post_count'].mean():.1f}",
                "Most Active Day (Replies)": peak_day_replies['day'],
                "Average Daily Replies": f"{daily_stats['avg_replies'].mean():.2f}"
            }

        for key, value in metrics.items():
            st.metric(key, value)


def get_network_data(start_date, end_date):
    """Get network data for Reddit AI discussions"""
    engine = init_connection()

    query = """
    WITH ai_posts AS (
        SELECT 
            p.post_id,
            p.subreddit,
            CAST(p.data->>'author' AS text) as post_author,
            CAST(p.data->>'score' AS integer) as post_score
        FROM reddit_posts p
        WHERE 
            CAST(FLOOR(CAST(p.data->>'created' AS float)) AS bigint) BETWEEN :start_date AND :end_date
            AND (
                LOWER(p.cleaned_title) LIKE '%artificial intelligence%'
                OR LOWER(p.cleaned_title) LIKE '%machine learning%'
                OR LOWER(p.cleaned_title) LIKE '%neural network%'
                OR LOWER(p.cleaned_title) LIKE '%deep learning%'
                OR LOWER(p.cleaned_title) LIKE '%ai %'
            )
    ),
    comment_interactions AS (
        SELECT 
            p.post_id,
            p.subreddit,
            p.post_author,
            p.post_score,
            CAST(c.data->>'author' AS text) as comment_author,
            c.score as comment_score
        FROM ai_posts p
        JOIN reddit_comments_cleaned c ON p.post_id = c.post_id
        WHERE 
            CAST(c.data->>'author' AS text) NOT IN ('AutoModerator', '[deleted]', 'None', 'null')
            AND p.post_author NOT IN ('AutoModerator', '[deleted]', 'None', 'null')
    )
    SELECT * FROM comment_interactions
    """

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(
                text(query),
                conn,
                params={'start_date': start_date, 'end_date': end_date}
            )
            return df
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()


def create_subreddit_network(df, selected_subreddit):
    """Create network visualization for top 10 users in selected subreddit"""
    # Filter data for selected subreddit
    subreddit_df = df[df['subreddit'] == selected_subreddit]

    # Get top 10 users in this subreddit by combining post and comment activity
    user_stats = pd.DataFrame()

    # Aggregate post statistics
    post_stats = subreddit_df.groupby('post_author').agg({
        'post_id': 'count',  # Number of posts
        'post_score': 'sum',  # Total post score
    }).reset_index()
    post_stats.columns = ['user', 'post_count', 'post_score']

    # Aggregate comment statistics
    comment_stats = subreddit_df.groupby('comment_author').agg({
        'comment_score': 'sum'  # Total comment score
    }).reset_index()
    comment_stats.columns = ['user', 'comment_score']

    # Merge post and comment stats
    user_stats = post_stats.merge(comment_stats, on='user', how='outer').fillna(0)

    # Calculate engagement score
    user_stats['engagement_score'] = (
            user_stats['post_score'] * 0.4 +  # 40% weight to post scores
            user_stats['comment_score'] * 0.3 +  # 30% weight to comment scores
            user_stats['post_count'] * 50 * 0.3  # 30% weight to number of posts
    ).astype(float)

    # Get top 10 users
    top_10_users = set(user_stats.nlargest(10, 'engagement_score')['user'].values)

    # Create network graph
    G = nx.Graph()

    # Track interactions between top users
    interactions = defaultdict(int)

    for _, row in subreddit_df.iterrows():
        if row['post_author'] in top_10_users and row['comment_author'] in top_10_users:
            if row['post_author'] != row['comment_author']:
                pair = tuple(sorted([row['post_author'], row['comment_author']]))
                interactions[pair] += 1

    # Add nodes and edges
    for user in top_10_users:
        G.add_node(user)

    for (user1, user2), weight in interactions.items():
        G.add_edge(user1, user2, weight=weight)

    # Calculate layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Create visualization traces
    edge_x, edge_y, edge_text = [], [], []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f'Interactions: {edge[2]["weight"]}')

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(150,150,150,0.5)'),
        hoverinfo='text',
        text=edge_text,
        mode='lines'
    )

    node_x, node_y, node_text, node_size = [], [], [], []

    # Get the maximum engagement score for scaling
    max_engagement = user_stats['engagement_score'].max()
    min_engagement = user_stats['engagement_score'].min()

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        user_data = user_stats[user_stats['user'] == node].iloc[0]
        node_text.append(
            f'User: {node}<br>' +
            f'Posts: {int(user_data["post_count"])}<br>' +
            f'Post Score: {int(user_data["post_score"])}<br>' +
            f'Comment Score: {int(user_data["comment_score"])}'
        )

        # New scaled node size calculation
        # Normalize engagement score to range [0,1] and then scale to reasonable size range
        if max_engagement != min_engagement:
            normalized_score = (user_data['engagement_score'] - min_engagement) / (max_engagement - min_engagement)
            node_size.append(20 + normalized_score * 25)  # Size range from 20 to 45
        else:
            node_size.append(30)  # Default size if all scores are equal

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=list(G.nodes()),
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color='#63B3ED',
            line=dict(width=1.5, color='white'),  # Reduced line width
            symbol='circle'
        ),
        textfont=dict(
            size=10,  # Slightly reduced text size
            color='white'
        )
    )

    # Update layout for better spacing
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f"Top 10 Users Interaction Network in r/{selected_subreddit}",
                y=0.95
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),  # Fixed range
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),  # Fixed range
            template="plotly_dark",
            height=800,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    )

    stats = {
        'user_stats': user_stats.nlargest(10, 'engagement_score'),
        'total_interactions': sum(interactions.values()),
        'avg_interactions': sum(interactions.values()) / len(interactions) if interactions else 0,
        'subreddit_stats': {
            'total_posts': len(subreddit_df['post_id'].unique()),
            'total_comments': len(subreddit_df),
            'active_users': len(set(subreddit_df['post_author']) | set(subreddit_df['comment_author']))
        }
    }

    return fig, stats


def display_network_analysis(df):
    """Display network analysis with subreddit selection"""
    st.header("🔄 Subreddit Network Analysis")

    # Get list of subreddits and their post counts
    subreddit_counts = df.groupby('subreddit').agg({
        'post_id': 'nunique'
    }).reset_index()
    subreddit_counts.columns = ['subreddit', 'post_count']
    subreddit_counts = subreddit_counts.sort_values('post_count', ascending=False)

    # Create subreddit selector
    selected_subreddit = st.selectbox(
        "Select Subreddit",
        subreddit_counts['subreddit'].tolist(),
        format_func=lambda
            x: f"r/{x} ({subreddit_counts[subreddit_counts['subreddit'] == x].iloc[0]['post_count']} posts)"
    )

    if selected_subreddit:
        fig, stats = create_subreddit_network(df, selected_subreddit)

        # Display subreddit overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Posts", stats['subreddit_stats']['total_posts'])
        with col2:
            st.metric("Total Comments", stats['subreddit_stats']['total_comments'])
        with col3:
            st.metric("Active Users", stats['subreddit_stats']['active_users'])

        # Display network visualization
        st.plotly_chart(fig, use_container_width=True)

        # Display user statistics
        st.subheader(f"Top Users in r/{selected_subreddit}")

        # Format statistics table
        stats_df = stats['user_stats'].copy()
        display_columns = ['user', 'post_count', 'post_score', 'comment_score', 'engagement_score']
        stats_df = stats_df[display_columns].round(2)
        stats_df.columns = ['User', 'Posts', 'Post Score', 'Comment Score', 'Engagement Score']

        st.dataframe(
            stats_df,
            hide_index=True,
            column_config={
                "User": st.column_config.TextColumn("User", width="medium"),
                "Posts": st.column_config.NumberColumn("Posts", format="%d"),
                "Post Score": st.column_config.NumberColumn("Post Score", format="%d"),
                "Comment Score": st.column_config.NumberColumn("Comment Score", format="%d"),
                "Engagement Score": st.column_config.NumberColumn("Engagement Score", format="%.2f")
            }
        )

@st.cache_data(ttl=3600)
def get_top_ai_users_reddit(start_date, end_date):
    """Get top users engaging in AI discussions on Reddit"""
    engine = init_connection()
    cache_key = get_cache_key('reddit_ai_users', start_date, end_date)

    cached_data = load_cache(cache_key)
    if cached_data is not None:
        return cached_data

    query = """
    WITH ai_posts AS (
        SELECT 
            p.post_id,
            p.subreddit,
            CAST(p.data->>'author' AS text) as author,
            CAST(p.data->>'score' AS integer) as post_score,
            COUNT(c.id) as comment_count,
            SUM(COALESCE(c.score, 0)) as total_comment_score
        FROM reddit_posts p
        LEFT JOIN reddit_comments_cleaned c ON p.post_id = c.post_id
        WHERE 
            CAST(FLOOR(CAST(p.data->>'created' AS float)) AS bigint) BETWEEN :start_date AND :end_date
            AND (
                LOWER(p.cleaned_title) LIKE '%artificial intelligence%'
                OR LOWER(p.cleaned_title) LIKE '%machine learning%'
                OR LOWER(p.cleaned_title) LIKE '%neural network%'
                OR LOWER(p.cleaned_title) LIKE '%deep learning%'
                OR LOWER(p.cleaned_title) LIKE '%ai %'
                OR LOWER(p.cleaned_text) LIKE '%artificial intelligence%'
                OR LOWER(p.cleaned_text) LIKE '%machine learning%'
                OR LOWER(p.cleaned_text) LIKE '%neural network%'
                OR LOWER(p.cleaned_text) LIKE '%deep learning%'
                OR LOWER(p.cleaned_text) LIKE '%ai %'
            )
        GROUP BY p.post_id, p.subreddit, p.data->>'author', CAST(p.data->>'score' AS integer)
    )
    SELECT 
        author,
        subreddit,
        COUNT(DISTINCT post_id) as post_count,
        SUM(post_score) as total_post_score,
        SUM(comment_count) as total_comments,
        SUM(total_comment_score) as total_comment_score,
        (SUM(post_score) + SUM(total_comment_score)) as total_engagement_score
    FROM ai_posts
    WHERE author NOT IN ('AutoModerator', '[deleted]', 'None', 'null')
    GROUP BY author, subreddit
    ORDER BY total_engagement_score DESC
    """

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(
                text(query),
                conn,
                params={'start_date': start_date, 'end_date': end_date}
            )
            save_cache(cache_key, df)
            return df
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()


def display_ai_engagement_analysis(df):
    """Display AI engagement analysis visualizations"""
    st.header("🤖 AI Discussion Engagement Analysis")

    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Active Users", len(df['author'].unique()))
    with col2:
        st.metric("Total Posts", df['post_count'].sum())
    with col3:
        st.metric("Total Comments", df['total_comments'].sum())
    with col4:
        st.metric("Avg Engagement Score", round(df['total_engagement_score'].mean(), 2))

    # Overall top users visualization
    st.subheader("Overall Top Users by Total Engagement")
    top_users = df.groupby('author').agg({
        'total_post_score': 'sum',
        'total_comment_score': 'sum',
        'total_engagement_score': 'sum'
    }).reset_index().nlargest(10, 'total_engagement_score')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_users['author'],
        y=top_users['total_post_score'],
        name='Post Score',
        marker_color='rgb(55, 83, 109)'
    ))
    fig.add_trace(go.Bar(
        x=top_users['author'],
        y=top_users['total_comment_score'],
        name='Comment Score',
        marker_color='rgb(26, 118, 255)'
    ))

    fig.update_layout(
        barmode='stack',
        title="Top 10 Users by Overall Engagement Score",
        xaxis_title="User",
        yaxis_title="Engagement Score",
        template="plotly_dark",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top users by subreddit
    st.subheader("Top Users by Subreddit")

    # Get top subreddits by total engagement
    subreddit_totals = df.groupby('subreddit')['total_engagement_score'].sum().sort_values(ascending=False)
    top_subreddits = subreddit_totals.head(10).index.tolist()

    # Create selectbox for subreddit selection
    selected_subreddit = st.selectbox(
        "Select Subreddit",
        top_subreddits,
        format_func=lambda x: f"r/{x}"
    )

    if selected_subreddit:
        # Get top users for selected subreddit
        subreddit_data = df[df['subreddit'] == selected_subreddit]
        subreddit_top_users = subreddit_data.nlargest(5, 'total_engagement_score')

        # Display metrics for this subreddit
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Users", len(subreddit_data['author'].unique()))
        with col2:
            st.metric("Total Posts", subreddit_data['post_count'].sum())
        with col3:
            st.metric("Total Comments", subreddit_data['total_comments'].sum())

        # Create visualization for top users in subreddit
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=subreddit_top_users['author'],
            y=subreddit_top_users['total_post_score'],
            name='Post Score',
            marker_color='rgb(55, 83, 109)'
        ))
        fig.add_trace(go.Bar(
            x=subreddit_top_users['author'],
            y=subreddit_top_users['total_comment_score'],
            name='Comment Score',
            marker_color='rgb(26, 118, 255)'
        ))

        fig.update_layout(
            barmode='stack',
            title=f"Top 5 Users in r/{selected_subreddit}",
            xaxis_title="User",
            yaxis_title="Engagement Score",
            template="plotly_dark",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show detailed user stats for this subreddit
        st.subheader(f"Detailed User Stats for r/{selected_subreddit}")
        user_stats = subreddit_data.nlargest(10, 'total_engagement_score')[
            ['author', 'post_count', 'total_post_score', 'total_comments',
             'total_comment_score', 'total_engagement_score']
        ].reset_index(drop=True)

        # Format the columns for better readability
        user_stats = user_stats.rename(columns={
            'author': 'User',
            'post_count': 'Posts',
            'total_post_score': 'Post Score',
            'total_comments': 'Comments',
            'total_comment_score': 'Comment Score',
            'total_engagement_score': 'Total Engagement'
        })

        st.dataframe(user_stats, use_container_width=True)

    # Add option to show full data table
    if st.checkbox("Show full dataset"):
        st.dataframe(df)

def get_network_data(start_date, end_date):
    """Get network data for Reddit AI discussions"""
    engine = init_connection()

    query = """
    WITH ai_posts AS (
        SELECT 
            p.post_id,
            p.subreddit,
            CAST(p.data->>'author' AS text) as post_author,
            CAST(p.data->>'score' AS integer) as post_score
        FROM reddit_posts p
        WHERE 
            CAST(FLOOR(CAST(p.data->>'created' AS float)) AS bigint) BETWEEN :start_date AND :end_date
            AND (
                LOWER(p.cleaned_title) LIKE '%artificial intelligence%'
                OR LOWER(p.cleaned_title) LIKE '%machine learning%'
                OR LOWER(p.cleaned_title) LIKE '%neural network%'
                OR LOWER(p.cleaned_title) LIKE '%deep learning%'
                OR LOWER(p.cleaned_title) LIKE '%ai %'
            )
    ),
    comment_interactions AS (
        SELECT 
            p.post_id,
            p.subreddit,
            p.post_author,
            p.post_score,
            CAST(c.data->>'author' AS text) as comment_author,
            c.score as comment_score
        FROM ai_posts p
        JOIN reddit_comments_cleaned c ON p.post_id = c.post_id
        WHERE 
            CAST(c.data->>'author' AS text) NOT IN ('AutoModerator', '[deleted]', 'None', 'null')
            AND p.post_author NOT IN ('AutoModerator', '[deleted]', 'None', 'null')
    )
    SELECT * FROM comment_interactions
    """

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(
                text(query),
                conn,
                params={'start_date': start_date, 'end_date': end_date}
            )
            return df
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()


def create_subreddit_network(df, selected_subreddit):
    """Create network visualization for top 10 users in selected subreddit"""
    # Filter data for selected subreddit
    subreddit_df = df[df['subreddit'] == selected_subreddit]

    # Get top 10 users in this subreddit by combining post and comment activity
    user_stats = pd.DataFrame()

    # Aggregate post statistics
    post_stats = subreddit_df.groupby('post_author').agg({
        'post_id': 'count',  # Number of posts
        'post_score': 'sum',  # Total post score
    }).reset_index()
    post_stats.columns = ['user', 'post_count', 'post_score']

    # Aggregate comment statistics
    comment_stats = subreddit_df.groupby('comment_author').agg({
        'comment_score': 'sum'  # Total comment score
    }).reset_index()
    comment_stats.columns = ['user', 'comment_score']

    # Merge post and comment stats
    user_stats = post_stats.merge(comment_stats, on='user', how='outer').fillna(0)

    # Calculate engagement score
    user_stats['engagement_score'] = (
            user_stats['post_score'] * 0.4 +  # 40% weight to post scores
            user_stats['comment_score'] * 0.3 +  # 30% weight to comment scores
            user_stats['post_count'] * 50 * 0.3  # 30% weight to number of posts
    ).astype(float)

    # Get top 10 users
    top_10_users = set(user_stats.nlargest(10, 'engagement_score')['user'].values)

    # Create network graph
    G = nx.Graph()

    # Track interactions between top users
    interactions = defaultdict(int)

    for _, row in subreddit_df.iterrows():
        if row['post_author'] in top_10_users and row['comment_author'] in top_10_users:
            if row['post_author'] != row['comment_author']:
                pair = tuple(sorted([row['post_author'], row['comment_author']]))
                interactions[pair] += 1

    # Add nodes and edges
    for user in top_10_users:
        G.add_node(user)

    for (user1, user2), weight in interactions.items():
        G.add_edge(user1, user2, weight=weight)

    # Calculate layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Create visualization traces
    edge_x, edge_y, edge_text = [], [], []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f'Interactions: {edge[2]["weight"]}')

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(150,150,150,0.5)'),
        hoverinfo='text',
        text=edge_text,
        mode='lines'
    )

    node_x, node_y, node_text, node_size = [], [], [], []

    # Get the maximum engagement score for scaling
    max_engagement = user_stats['engagement_score'].max()
    min_engagement = user_stats['engagement_score'].min()

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        user_data = user_stats[user_stats['user'] == node].iloc[0]
        node_text.append(
            f'User: {node}<br>' +
            f'Posts: {int(user_data["post_count"])}<br>' +
            f'Post Score: {int(user_data["post_score"])}<br>' +
            f'Comment Score: {int(user_data["comment_score"])}'
        )

        # New scaled node size calculation
        # Normalize engagement score to range [0,1] and then scale to reasonable size range
        if max_engagement != min_engagement:
            normalized_score = (user_data['engagement_score'] - min_engagement) / (max_engagement - min_engagement)
            node_size.append(20 + normalized_score * 25)  # Size range from 20 to 45
        else:
            node_size.append(30)  # Default size if all scores are equal

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=list(G.nodes()),
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color='#63B3ED',
            line=dict(width=1.5, color='white'),  # Reduced line width
            symbol='circle'
        ),
        textfont=dict(
            size=10,  # Slightly reduced text size
            color='white'
        )
    )

    # Update layout for better spacing
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f"Top 10 Users Interaction Network in r/{selected_subreddit}",
                y=0.95
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),  # Fixed range
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.2]),  # Fixed range
            template="plotly_dark",
            height=800,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    )

    stats = {
        'user_stats': user_stats.nlargest(10, 'engagement_score'),
        'total_interactions': sum(interactions.values()),
        'avg_interactions': sum(interactions.values()) / len(interactions) if interactions else 0,
        'subreddit_stats': {
            'total_posts': len(subreddit_df['post_id'].unique()),
            'total_comments': len(subreddit_df),
            'active_users': len(set(subreddit_df['post_author']) | set(subreddit_df['comment_author']))
        }
    }

    return fig, stats


def display_network_analysis(df):
    """Display network analysis with subreddit selection"""
    st.header("🔄 Subreddit Network Analysis")

    # Get list of subreddits and their post counts
    subreddit_counts = df.groupby('subreddit').agg({
        'post_id': 'nunique'
    }).reset_index()
    subreddit_counts.columns = ['subreddit', 'post_count']
    subreddit_counts = subreddit_counts.sort_values('post_count', ascending=False)

    # Create subreddit selector
    selected_subreddit = st.selectbox(
        "Select Subreddit",
        subreddit_counts['subreddit'].tolist(),
        format_func=lambda
            x: f"r/{x} ({subreddit_counts[subreddit_counts['subreddit'] == x].iloc[0]['post_count']} posts)"
    )

    if selected_subreddit:
        fig, stats = create_subreddit_network(df, selected_subreddit)

        # Display subreddit overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Posts", stats['subreddit_stats']['total_posts'])
        with col2:
            st.metric("Total Comments", stats['subreddit_stats']['total_comments'])
        with col3:
            st.metric("Active Users", stats['subreddit_stats']['active_users'])

        # Display network visualization
        st.plotly_chart(fig, use_container_width=True)

        # Display user statistics
        st.subheader(f"Top Users in r/{selected_subreddit}")

        # Format statistics table
        stats_df = stats['user_stats'].copy()
        display_columns = ['user', 'post_count', 'post_score', 'comment_score', 'engagement_score']
        stats_df = stats_df[display_columns].round(2)
        stats_df.columns = ['User', 'Posts', 'Post Score', 'Comment Score', 'Engagement Score']

        st.dataframe(
            stats_df,
            hide_index=True,
            column_config={
                "User": st.column_config.TextColumn("User", width="medium"),
                "Posts": st.column_config.NumberColumn("Posts", format="%d"),
                "Post Score": st.column_config.NumberColumn("Post Score", format="%d"),
                "Comment Score": st.column_config.NumberColumn("Comment Score", format="%d"),
                "Engagement Score": st.column_config.NumberColumn("Engagement Score", format="%.2f")
            }
        )

def display_hourly_patterns(hourly_stats, platform):
    """Display hourly patterns visualization"""
    if platform == "Reddit":
        fig = px.line(hourly_stats,
                      x='hour',
                      y=['post_count', 'comment_count'],
                      title="Average Hourly Activity",
                      labels={'hour': 'Hour of Day (UTC)',
                              'value': 'Count',
                              'variable': 'Metric'},
                      template="plotly_dark")
    else:
        fig = px.line(hourly_stats,
                      x='hour',
                      y=['post_count', 'avg_replies'],
                      title="Average Hourly Activity",
                      labels={'hour': 'Hour of Day (UTC)',
                              'value': 'Count',
                              'variable': 'Metric'},
                      template="plotly_dark")

    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(
        tickmode='array',
        ticktext=[f'{h:02d}:00' for h in range(24)],
        tickvals=list(range(24)),
        tickangle=45,
        gridcolor='rgba(128,128,128,0.2)',
        title_standoff=25
    )
    fig.update_layout(
        xaxis_title="Hour of Day (UTC)",
        yaxis_title="Average Count",
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def display_daily_patterns(daily_stats, platform):
    """Display daily patterns visualization"""
    if platform == "Reddit":
        fig = px.bar(daily_stats,
                     x='day',
                     y=['post_count', 'comment_count'],
                     title="Average Daily Activity",
                     labels={'day': 'Day of Week',
                             'value': 'Count',
                             'variable': 'Metric'},
                     template="plotly_dark",
                     barmode='group')
    else:
        fig = px.bar(daily_stats,
                     x='day',
                     y=['post_count', 'avg_replies'],
                     title="Average Daily Activity",
                     labels={'day': 'Day of Week',
                             'value': 'Count',
                             'variable': 'Metric'},
                     template="plotly_dark",
                     barmode='group')

    fig.update_layout(
        xaxis_title="Day of Week",
        yaxis_title="Average Count",
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    st.plotly_chart(fig, use_container_width=True)