from sqlalchemy import create_engine, text
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter

def plot_politics_submissions(db_url, start_date='2024-11-01', end_date='2024-11-14'):
    try:
        engine = create_engine(db_url)

        query = text("""
        SELECT 
            DATE(to_timestamp(CAST((data->>'created') AS FLOAT8))) as post_date,
            COUNT(*) as post_count
        FROM reddit_posts
        WHERE 
            subreddit = 'politics'
            AND DATE(to_timestamp(CAST((data->>'created') AS FLOAT8))) BETWEEN :start_date AND :end_date
        GROUP BY DATE(to_timestamp(CAST((data->>'created') AS FLOAT8)))
        ORDER BY post_date;
        """)

        with engine.connect() as connection:
            df = pd.read_sql_query(
                query,
                connection,
                params={
                    'start_date': start_date,
                    'end_date': end_date
                }
            )

        if df.empty:
            print("No data found for r/politics in the specified date range.")
            return None

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        df['post_date'] = pd.to_datetime(df['post_date']).dt.date
        date_df = pd.DataFrame({'post_date': [d.date() for d in date_range]})
        df = pd.merge(date_df, df, on='post_date', how='left')
        df['post_date'] = pd.to_datetime(df['post_date'])
        df['post_count'] = df['post_count'].fillna(0).astype(int)

        # Enhanced plot settings
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(16, 8))  # Increased figure size
        
        # Set background colors
        ax.set_facecolor('#f8f9fa')  # Light gray background
        fig.patch.set_facecolor('white')
        
        # Plot line with enhanced styling
        line = ax.plot(
            df['post_date'],
            df['post_count'],
            marker='o',
            linestyle='-',
            linewidth=3,
            color='#2196F3',  # Brighter blue
            markersize=10,
            label='r/politics'
        )

        # Add value labels with improved positioning and style
        for x, y in zip(df['post_date'], df['post_count']):
            if y >= 0:
                ax.annotate(
                    str(int(y)),
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 15),  # Increased vertical offset
                    ha='center',
                    fontsize=11,  # Larger font
                    color='#1976D2',  # Darker blue for better contrast
                    weight='bold',
                    bbox=dict(
                        facecolor='white',
                        edgecolor='none',
                        alpha=0.7,
                        pad=1
                    )
                )

        # Enhanced title and labels
        ax.set_title('Daily Submissions in r/politics', 
                    fontsize=16, 
                    pad=20, 
                    weight='bold')
        ax.set_xlabel('Date', fontsize=14, labelpad=10)
        ax.set_ylabel('Number of Submissions', fontsize=14, labelpad=10)
        
        # Enhanced grid
        ax.grid(True, linestyle='--', alpha=0.3, color='gray', which='major')
        
        # Improved x-axis formatting
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(date_range, rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        
        # Set axis limits with padding
        ax.set_xlim(date_range[0] - timedelta(days=0.5), 
                   date_range[-1] + timedelta(days=0.5))
        
        # Add padding to y-axis
        ymax = df['post_count'].max()
        ax.set_ylim(0, ymax * 1.15)  # 15% padding on top

        # Enhanced summary statistics box
        stats_text = (
            f"Summary Statistics:\n\n"
            f"Total Submissions: {df['post_count'].sum():,}\n"
            f"Average: {df['post_count'].mean():.1f} posts/day\n"
            f"Maximum: {df['post_count'].max()} posts/day\n"
            f"Minimum: {df['post_count'].min()} posts/day"
        )

        plt.figtext(
            1.12, 0.5,
            stats_text,
            bbox=dict(
                facecolor='white',
                edgecolor='#ddd',  # Light gray border
                alpha=1.0,
                pad=1.5,
                boxstyle='round,pad=1'  # Rounded corners
            ),
            fontsize=12  # Larger font
        )

        # Adjusted layout with more space for the statistics box
        plt.subplots_adjust(right=0.85, bottom=0.2, left=0.1, top=0.9)

        # Add border to plot
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#ddd')

        return fig

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Example usage:
if __name__ == "__main__":
    db_url = 'postgresql://postgres:password@localhost:5432/crawler_db'
    fig = plot_politics_submissions(db_url)
    if fig:
        plt.show()