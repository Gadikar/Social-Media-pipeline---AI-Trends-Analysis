from sqlalchemy import create_engine, text
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import seaborn as sns

def plot_subreddit_activity(db_url, subreddits, days_back=30):
    """
    Create a line graph showing daily post counts for specified subreddits.
    Enhanced visibility for all subreddit lines.
    """
    try:
        # Create SQLAlchemy engine
        engine = create_engine(db_url)

        query = text("""
        SELECT 
            DATE(to_timestamp((data->>'created')::float)) as post_date,
            subreddit,
            COUNT(*) as post_count
        FROM reddit_posts
        WHERE 
            subreddit = ANY(:subreddits)
            AND to_timestamp((data->>'created')::float) >= NOW() - INTERVAL ':days_back days'
        GROUP BY DATE(to_timestamp((data->>'created')::float)), subreddit
        ORDER BY post_date;
        """)

        with engine.connect() as connection:
            df = pd.read_sql_query(
                query,
                connection,
                params={
                    'subreddits': subreddits,
                    'days_back': days_back
                }
            )

        if df.empty:
            print("No data found for the specified subreddits and time range.")
            return None

        # Set up the plot with white background
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Define distinct colors and styles for each subreddit
        style_config = {
            'artificial': {
                'color': '#FF6B6B',  # Coral red
                'marker': 'o',
                'linestyle': '-',
                'linewidth': 2.5,
                'markersize': 8
            },
            'OpenAI': {
                'color': '#4ECDC4',  # Turquoise
                'marker': 's',
                'linestyle': '-',
                'linewidth': 2.5,
                'markersize': 8
            },
            'localLLaMA': {
                'color': '#45B7D1',  # Bright blue
                'marker': '^',
                'linestyle': '-',
                'linewidth': 3,
                'markersize': 10
            }
        }

        # Plot line for each subreddit
        for subreddit, group in df.groupby('subreddit'):
            dates = pd.to_datetime(group['post_date'])
            style = style_config.get(subreddit, {
                'color': '#666666',
                'marker': 'o',
                'linestyle': '-',
                'linewidth': 2,
                'markersize': 8
            })
            
            # Plot main line
            ax.plot(
                dates,
                group['post_count'],
                label=f'r/{subreddit}',
                **style,
                alpha=0.8
            )
            
            # Add value labels with matching colors
            for x, y in zip(dates, group['post_count']):
                ax.annotate(
                    str(y),
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=9,
                    color=style['color'],
                    weight='bold'
                )

        # Customize the plot
        ax.set_title('Daily Posts per Subreddit', fontsize=14, pad=20, weight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Number of Posts', fontsize=12)
        
        # Add grid with light gray color
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # Add legend with total posts and better visibility
        total_posts = df.groupby('subreddit')['post_count'].sum()
        legend_labels = [
            f'r/{sub} (Total: {total_posts[sub]:,})' 
            for sub in total_posts.index
        ]
        ax.legend(
            legend_labels, 
            bbox_to_anchor=(1.05, 1), 
            loc='upper left',
            frameon=True,
            facecolor='white',
            edgecolor='gray'
        )

        # Format x-axis dates
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        # Add summary statistics text box with improved visibility
        stats_text = "Summary Statistics:\n"
        for subreddit in df['subreddit'].unique():
            subreddit_data = df[df['subreddit'] == subreddit]['post_count']
            style = style_config.get(subreddit, {'color': '#666666'})
            stats_text += f"\nr/{subreddit} "
            stats_text += f"(shown in {style['color']}):\n"
            stats_text += f"Avg: {subreddit_data.mean():.1f} posts/day\n"
            stats_text += f"Max: {subreddit_data.max()} posts/day\n"
            stats_text += f"Min: {subreddit_data.min()} posts/day"

        plt.figtext(
            1.25, 0.5, 
            stats_text,
            bbox=dict(
                facecolor='white',
                edgecolor='gray',
                alpha=1.0,
                pad=1.0
            ),
            fontsize=9
        )

        # Adjust layout
        plt.subplots_adjust(right=0.85, bottom=0.15)

        return fig

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Example usage:
if __name__ == "__main__":
    db_url = 'postgresql://postgres:password@localhost:5433/crawler_db'
    subreddits = ['artificial', 'OpenAI', 'localLLama']
    fig = plot_subreddit_activity(db_url, subreddits)
    if fig:
        plt.show()

#postgresql://postgres:password@localhost:5433/crawler_db
       