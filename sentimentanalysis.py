'''
import pandas as pd
import plotly.graph_objects as go
from textblob import TextBlob
import time
from datetime import datetime
import json

def calculate_sentiment(text):
    """Calculate sentiment score for given text"""
    if pd.isna(text) or text == '' or not isinstance(text, str):
        return None
    try:
        return round(TextBlob(str(text)).sentiment.polarity, 2)
    except:
        return None

def extract_created_timestamp(json_str):
    """Extract created timestamp from JSON data"""
    try:
        data = json.loads(json_str)
        return data.get('created')
    except:
        return None

def process_posts_sentiment(csv_file):
    """
    Process Reddit posts sentiment from CSV file
    
    Args:
        csv_file (str): Path to the CSV file containing Reddit posts data
    """
    print("Processing Reddit posts data...")
    
    # Define required columns
    required_columns = ['post_id', 'subreddit', 'data', 'cleaned_title']
    
    # Initialize empty list for chunks
    chunks = []
    chunk_size = 10000
    
    # Calculate date range
    start_date = int(datetime(2024, 11, 1).timestamp())
    end_date = int(datetime(2024, 11, 30).timestamp())
    
    print(f"Reading data from {csv_file} in chunks...")
    
    try:
        for chunk_num, chunk in enumerate(pd.read_csv(csv_file, usecols=required_columns, chunksize=chunk_size), 1):
            print(f"\nProcessing chunk {chunk_num}...")
            
            # Extract created timestamp from data column
            chunk['created_at'] = chunk['data'].apply(extract_created_timestamp)
            
            # Convert created_at to numeric and handle NaN values
            chunk['created_at'] = pd.to_numeric(chunk['created_at'], errors='coerce')
            
            # Filter data for November 2024
            chunk = chunk[
                (chunk['created_at'].notna()) &
                (chunk['created_at'] >= start_date) & 
                (chunk['created_at'] <= end_date) &
                (~chunk['cleaned_title'].isna())
            ]
            
            if chunk.empty:
                continue
                
            # Calculate sentiment scores
            chunk['sentiment_score'] = chunk['cleaned_title'].apply(calculate_sentiment)
            
            # Drop rows where sentiment_score is None
            chunk = chunk.dropna(subset=['sentiment_score'])
            
            # Convert timestamp to datetime
            chunk['date'] = pd.to_datetime(chunk['created_at'], unit='s')
            
            chunks.append(chunk[['date', 'sentiment_score', 'subreddit']])
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    if not chunks:
        print("No data found for the specified date range!")
        return
    
    # Combine all chunks
    df = pd.concat(chunks, ignore_index=True)
    
    # Calculate daily metrics
    all_dates = pd.date_range(datetime(2024, 11, 1), datetime(2024, 11, 30), freq='D').strftime('%Y-%m-%d')
    
    daily_sentiment = []
    
    for date in all_dates:
        date_data = df[df['date'].dt.date == pd.to_datetime(date).date()]
        
        if not date_data.empty:
            avg_sentiment = round(date_data['sentiment_score'].mean(), 2)
            total_posts = len(date_data)
        else:
            avg_sentiment = 0
            total_posts = 0
            
        daily_sentiment.append({
            'date': date,
            'avg_sentiment': avg_sentiment,
            'total_posts': total_posts
        })
    
    daily_sentiment = pd.DataFrame(daily_sentiment)
    
    # Save daily metrics to CSV
    output_file = 'daily_sentiment_metrics.csv'
    daily_sentiment.to_csv(output_file, index=False)
    print(f"\nDaily metrics saved to {output_file}")
    
    # Print the daily metrics
    print("\nDaily Average Sentiment Scores for Reddit Posts:")
    print("Date | Average Sentiment | Total Posts")
    print("-" * 50)
    for _, row in daily_sentiment.iterrows():
        print(f"{row['date']} | {row['avg_sentiment']:.2f} | {row['total_posts']}")
    
    # Create the plot
    fig = go.Figure()
    
    # Add daily average sentiment line
    fig.add_trace(
        go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['avg_sentiment'],
            mode='lines+markers',
            name='Daily Average Sentiment',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        )
    )
    
    fig.update_layout(
        title={
            'text': 'Daily Average Reddit Posts Sentiment - November 2024',
            'y':0.95,
            'x':0.05,
            'xanchor': 'left',
            'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        plot_bgcolor='white',
        showlegend=True,
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            dtick="D2",
            tickformat="%b %d\n%Y",
            tickangle=45
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zerolinecolor='lightgray',
            zerolinewidth=1,
            range=[-1, 1],
            tickformat='.2f'
        ),
        height=600
    )
    
    # Save and display the figure
    fig.write_html("reddit_posts_daily_sentiment.html")
    fig.show()

if __name__ == "__main__":
    start_time = time.time()
    process_posts_sentiment('reddit_posts_export.csv')  # Provide your CSV filename here
    print(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")

    '''

#4chan 

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob
import time
from datetime import datetime, timedelta

def calculate_sentiment(text):
    """Calculate sentiment score for given text"""
    if pd.isna(text) or text == '' or not isinstance(text, str):
        return None
    try:
        # Round the sentiment score to 2 decimal places
        return round(TextBlob(str(text)).sentiment.polarity, 2)
    except:
        return None

def process_posts_sentiment(posts_file='chan_posts.csv'):
    print("Reading posts data...")
    
    # Define the date range
    start_date = datetime(2024, 11, 1)
    end_date = datetime(2024, 11, 14)
    
    chunks = []
    chunk_num = 0
    chunk_size = 10000
    
    for chunk in pd.read_csv(posts_file, chunksize=chunk_size):
        chunk_num += 1
        print(f"\nProcessing chunk {chunk_num}...")
        
        # Filter only 'g' and 'pol' boards
        valid_posts = chunk[chunk['board'].isin(['g', 'pol'])]
        
        if not valid_posts.empty:
            valid_posts['date'] = pd.to_datetime(valid_posts['created_at'])
            valid_posts['sentiment_score'] = valid_posts['cleaned_text'].apply(calculate_sentiment)
            valid_posts = valid_posts.dropna(subset=['sentiment_score'])
            chunks.append(valid_posts[['date', 'sentiment_score', 'board']])
    
    df = pd.concat(chunks, ignore_index=True)
    
    # Generate the complete date range
    all_dates = pd.date_range(start_date, end_date, freq='D').strftime('%Y-%m-%d')
    
    daily_sentiment = []
    cumulative_sentiment = 0
    
    for date in all_dates:
        date_data = df[df['date'].dt.date == pd.to_datetime(date).date()]
        
        if not date_data.empty:
            # Round average sentiment to 2 decimal places
            avg_sentiment = round(date_data['sentiment_score'].mean(), 2)
            cumulative_sentiment = round(cumulative_sentiment + avg_sentiment, 2)  # Round cumulative as well
        else:
            avg_sentiment = 0
            
        daily_sentiment.append({
            'date': date,
            'avg_sentiment': avg_sentiment,
            'cumulative_sentiment': cumulative_sentiment
        })
    
    daily_sentiment = pd.DataFrame(daily_sentiment)
    
    # Print the daily metrics
    print("\nDaily and Cumulative Sentiment Scores for /g/ and /pol/ boards combined:")
    print("Date | Average Sentiment | Cumulative Sentiment")
    print("-" * 60)
    for _, row in daily_sentiment.iterrows():
        print(f"{row['date']} | {row['avg_sentiment']:.2f} | {row['cumulative_sentiment']:.2f}")
    
    # Create figure with subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Plot average sentiment - blue line
    fig.add_trace(
        go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['avg_sentiment'],
            mode='lines+markers',
            name='Daily Average Sentiment',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ),
        secondary_y=False
    )
    
    # Plot cumulative sentiment - red line
    fig.add_trace(
        go.Scatter(
            x=daily_sentiment['date'],
            y=daily_sentiment['cumulative_sentiment'],
            mode='lines+markers',
            name='Cumulative Sentiment',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=8)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title={
            'text': 'Sentiment Analysis (/g/ and /pol/ boards)',
            'y':0.95,
            'x':0.05,
            'xanchor': 'left',
            'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Daily Average Sentiment",
        yaxis2_title="Cumulative Sentiment",
        plot_bgcolor='white',
        legend=dict(
            title="Metric",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        showlegend=True,
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            dtick="D1",
            tickformat="%b %d\n%Y"
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zerolinecolor='lightgray',
            zerolinewidth=1,
            range=[-1, 1],
            tickformat='.2f'  # Format y-axis ticks to 2 decimal places
        ),
        yaxis2=dict(
            showgrid=False,
            zerolinecolor='lightgray',
            zerolinewidth=1,
            tickformat='.2f'  # Format secondary y-axis ticks to 2 decimal places
        )
    )
    
    # Save and display the figure
    fig.write_html("chan_posts_sentiment.html")
    fig.show()

if __name__ == "__main__":
    start_time = time.time()
    process_posts_sentiment('chan_posts.csv')
    print(f"\nProcessing completed in {time.time() - start_time:.2f} seconds")
    