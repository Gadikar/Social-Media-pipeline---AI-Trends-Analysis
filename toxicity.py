import pandas as pd
import psycopg2
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


def fetch_chan_data():
    # Replace with your actual database connection parameters
    conn = psycopg2.connect(
        dbname="crawler_db",
        user="postgres",
        password="password",
        host="localhost",
        port="5432"
    )

    query = """
    SELECT 
        DATE(to_timestamp(created_at)) as post_date,
        COUNT(*) as post_count,
        AVG(CASE WHEN class = 'flag' THEN score END) as avg_toxicity_score,
        SUM(CASE WHEN class = 'flag' THEN 1 ELSE 0 END) as toxic_posts_count
    FROM chan_posts
    GROUP BY DATE(to_timestamp(created_at))
    ORDER BY post_date;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def create_toxicity_visualization(df):
    # Create figure and axis with two subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot 1: Post Count and Toxic Posts Count
    ax1.plot(df['post_date'], df['post_count'], label='Total Posts', color='blue')
    ax1.plot(df['post_date'], df['toxic_posts_count'], label='Toxic Posts', color='red')
    ax1.set_title('Daily Post Counts')
    ax1.set_ylabel('Number of Posts')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Average Toxicity Score
    ax2.plot(df['post_date'], df['avg_toxicity_score'], label='Avg Toxicity Score', color='purple')
    ax2.set_title('Daily Average Toxicity Score')
    ax2.set_ylabel('Toxicity Score')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust layout to prevent label overlap
    plt.tight_layout()

    # Calculate and print some statistics
    print("Toxicity Analysis Summary:")
    print(f"Average daily toxicity score: {df['avg_toxicity_score'].mean():.3f}")
    print(f"Maximum daily toxicity score: {df['avg_toxicity_score'].max():.3f}")
    print(f"Total toxic posts: {df['toxic_posts_count'].sum()}")
    print(f"Percentage of toxic posts: {(df['toxic_posts_count'].sum() / df['post_count'].sum() * 100):.2f}%")

    return fig


# Main execution
if __name__ == "__main__":
    df = fetch_chan_data()
    fig = create_toxicity_visualization(df)
    plt.show()