from sqlalchemy import create_engine, text
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

def plot_reddit_hourly_comments(db_url, start_date='2024-11-01', end_date='2024-11-14'):
    try:
        engine = create_engine(db_url)

        # Query to get hourly comment counts with continuous hours
        query = text("""
        WITH RECURSIVE 
        hour_series AS (
            SELECT to_timestamp(:start_date, 'YYYY-MM-DD') as hour
            UNION ALL
            SELECT hour + interval '1 hour'
            FROM hour_series
            WHERE hour < to_timestamp(:end_date, 'YYYY-MM-DD') + interval '1 day'
        ),
        hourly_counts AS (
            SELECT 
                date_trunc('hour', to_timestamp(CAST((data->>'created') AS FLOAT))) as timestamp,
                COUNT(*) as comment_count
            FROM reddit_comments_cleaned
            WHERE 
                to_timestamp(CAST((data->>'created') AS FLOAT))
                    BETWEEN to_timestamp(:start_date, 'YYYY-MM-DD') 
                    AND to_timestamp(:end_date, 'YYYY-MM-DD') + interval '1 day'
            GROUP BY timestamp
        )
        SELECT 
            hour_series.hour as timestamp,
            COALESCE(hourly_counts.comment_count, 0) as comment_count
        FROM hour_series
        LEFT JOIN hourly_counts ON hour_series.hour = hourly_counts.timestamp
        ORDER BY hour_series.hour;
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
            print("No data found for the specified time range.")
            return None

        # Create figure and axes
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(15, 8))

        # Create line plot
        ax.plot(
            df['timestamp'],
            df['comment_count'],
            linewidth=2,
            color='#1f77b4',
            alpha=0.8,
            label='Comments per Hour'
        )

        # Format x-axis to show dates every 12 hours
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:00'))

        # Add minor ticks for 6-hour intervals
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        
        # Add grid
        ax.grid(True, which='major', linestyle='-', alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', alpha=0.4)

        # Customize the plot
        ax.set_title('Hourly Reddit Comments\n(Nov 1 - Nov 14, 2024)', 
                    fontsize=14, 
                    pad=20)
        ax.set_xlabel('Date and Hour (MM-DD Hour:00)', fontsize=12)
        ax.set_ylabel('Number of Comments', fontsize=12)

        # Format tick labels
        ax.tick_params(axis='both', labelsize=10)
        plt.xticks(rotation=45, ha='right')

        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)

        # Calculate summary statistics
        total_comments = df['comment_count'].sum()
        avg_comments = df['comment_count'].mean()
        max_comments = df['comment_count'].max()
        min_comments = df['comment_count'].min()
        peak_time = df.loc[df['comment_count'].idxmax(), 'timestamp']

        stats_text = (
            f"Summary Statistics:\n\n"
            f"Total Comments: {total_comments:,}\n"
            f"Average: {avg_comments:.1f} comments/hour\n"
            f"Maximum: {max_comments} comments/hour\n"
            f"Minimum: {min_comments} comments/hour\n"
            f"Peak Time: {peak_time.strftime('%Y-%m-%d %H:00')}"
        )

        # Add text box with statistics
        plt.figtext(
            1.02, 0.6,
            stats_text,
            bbox=dict(
                facecolor='white',
                edgecolor='gray',
                alpha=1.0,
                pad=1.0
            ),
            fontsize=10
        )

        # Add legend
        ax.legend()

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)

        return fig

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Example usage:
if __name__ == "__main__":
    db_url = 'postgresql://postgres:password@localhost:5432/crawler_db'
    fig = plot_reddit_hourly_comments(db_url, '2024-11-01', '2024-11-14')
    if fig:
        plt.show()