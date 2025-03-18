import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns

def extract_ups_from_data(row):
    """Extract ups value from JSON data string"""
    try:
        if isinstance(row, str):
            data = json.loads(row)
        else:
            data = row
        return data.get('ups', 0)
    except:
        return 0

def extract_created_from_data(row):
    """Extract created timestamp from JSON data string"""
    try:
        if isinstance(row, str):
            data = json.loads(row)
        else:
            data = row
        return data.get('created', 0)
    except:
        return 0

def plot_upvotes_trend(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file, low_memory=False)
    
    # Extract ups and created timestamp from data column
    df['ups'] = df['data'].apply(extract_ups_from_data)
    df['created'] = df['data'].apply(extract_created_from_data)
    
    # Convert timestamp to datetime
    df['date'] = pd.to_datetime(df['created'], unit='s')
    
    # Calculate daily average upvotes
    daily_avg = df.groupby(df['date'].dt.date)['ups'].mean().reset_index()
    
    # Sort by date
    daily_avg = daily_avg.sort_values('date')
    
    # Keep only last 30 days
    last_30_days = daily_avg.tail(30)

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Plot the line
    plt.plot(last_30_days['date'], last_30_days['ups'], 
             marker='o', linestyle='-', color='green', 
             label='Ups')
    
    # Customize the plot
    plt.title('Reddit Average Ups Over Time')
    plt.xlabel('Day')
    plt.ylabel('Average Ups')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('reddit_upvotes_trend.png')
    plt.close()

# Usage
if __name__ == "__main__":
    plot_upvotes_trend('reddit_posts_export.csv')