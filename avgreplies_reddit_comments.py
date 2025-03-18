import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

def extract_from_data(row, field):
    """Extract a field from JSON data string"""
    try:
        if isinstance(row, str):
            data = json.loads(row)
        else:
            data = row
        return data.get(field, 0)
    except:
        return 0

def plot_replies_trend(comments_csv):
    # Read the CSV file
    df = pd.read_csv(comments_csv, low_memory=False)
    
    # Extract created timestamp and parent_id from data column
    df['created'] = df['data'].apply(lambda x: extract_from_data(x, 'created'))
    df['parent_id'] = df['data'].apply(lambda x: extract_from_data(x, 'parent_id'))
    
    # Convert timestamp to datetime
    df['date'] = pd.to_datetime(df['created'], unit='s')
    
    # Count replies per post per day
    daily_replies = df.groupby([
        df['date'].dt.date,
        'parent_id'
    ]).size().reset_index(name='reply_count')
    
    # Calculate average replies per post per day
    daily_avg_replies = daily_replies.groupby('date')['reply_count'].mean().reset_index()
    
    # Sort by date
    daily_avg_replies = daily_avg_replies.sort_values('date')
    
    # Keep only last 30 days
    last_30_days = daily_avg_replies.tail(30)

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Plot the line
    plt.plot(last_30_days['date'], last_30_days['reply_count'], 
             marker='o', linestyle='-', color='blue', 
             label='Average Replies')
    
    # Customize the plot
    plt.title('Reddit Average Replies Over Time')
    plt.xlabel('Day')
    plt.ylabel('Average Replies per Post')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('reddit_replies_trend.png')
    plt.close()

# Usage
if __name__ == "__main__":
    plot_replies_trend('reddit_comments_clean.csv')