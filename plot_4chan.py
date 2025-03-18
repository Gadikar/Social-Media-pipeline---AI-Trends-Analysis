from sqlalchemy import create_engine, text
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import seaborn as sns

def plot_4chan_analysis(db_url, days_back=30):
    """
    Create a combined visualization showing:
    1. Daily post totals as a line graph
    2. Hourly distribution as a heatmap
    
    Parameters:
    -----------
    db_url : str
        SQLAlchemy database URL
    days_back : int
        Number of days of data to analyze
    """
    try:
        engine = create_engine(db_url)

        # Query for daily totals
        daily_query = text("""
        WITH post_dates AS (
            SELECT 
                DATE(to_timestamp((data->>'time')::float)) as post_date
            FROM chan_posts
            WHERE 
                (data->>'resto')::integer = 0
                AND to_timestamp((data->>'time')::float) >= NOW() - INTERVAL ':days_back days'
        )
        SELECT 
            post_date,
            COUNT(*) as post_count
        FROM post_dates
        GROUP BY post_date
        ORDER BY post_date;
        """)

        # Query for hourly distribution
        hourly_query = text("""
        WITH post_times AS (
            SELECT 
                DATE(to_timestamp((data->>'time')::float)) as post_date,
                EXTRACT(HOUR FROM to_timestamp((data->>'time')::float)) as hour
            FROM chan_posts
            WHERE 
                (data->>'resto')::integer = 0
                AND to_timestamp((data->>'time')::float) >= NOW() - INTERVAL ':days_back days'
        )
        SELECT 
            post_date,
            hour,
            COUNT(*) as post_count
        FROM post_times
        GROUP BY post_date, hour
        ORDER BY post_date, hour;
        """)

        with engine.connect() as connection:
            df_daily = pd.read_sql_query(daily_query, connection, params={'days_back': days_back})
            df_hourly = pd.read_sql_query(hourly_query, connection, params={'days_back': days_back})

        if df_daily.empty or df_hourly.empty:
            print("No data found for the specified time range.")
            return None

        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

        # Daily totals plot (top)
        ax1 = fig.add_subplot(gs[0])
        
        # Convert dates to datetime
        df_daily['post_date'] = pd.to_datetime(df_daily['post_date'])

        # Plot the line graph
        ax1.plot(
            df_daily['post_date'],
            df_daily['post_count'],
            marker='o',
            linestyle='-',
            linewidth=2,
            color='#2E86C1',
            markersize=8,
            markerfacecolor='white',
            markeredgecolor='#2E86C1',
            markeredgewidth=2
        )

        # Add value labels
        for x, y in zip(df_daily['post_date'], df_daily['post_count']):
            ax1.annotate(
                str(int(y)),
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color='#2E86C1'
            )

        ax1.set_title('Daily Root Posts', fontsize=14, pad=20)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Number of Posts', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Heatmap (bottom)
        ax2 = fig.add_subplot(gs[1])
        
        # Pivot data for heatmap
        pivot_df = df_hourly.pivot(
            index='post_date',
            columns='hour',
            values='post_count'
        ).fillna(0)

        # Create heatmap
        sns.heatmap(
            pivot_df,
            cmap='YlOrRd',
            ax=ax2,
            cbar_kws={'label': 'Number of Posts'},
            fmt='.0f',
            annot=True,
            annot_kws={'size': 8}
        )

        ax2.set_title('Hourly Post Distribution', fontsize=14, pad=20)
        ax2.set_xlabel('Hour of Day (UTC)', fontsize=12)
        ax2.set_ylabel('Date', fontsize=12)

        # Calculate statistics
        total_posts = df_daily['post_count'].sum()
        avg_posts = df_daily['post_count'].mean()
        max_posts = df_daily['post_count'].max()
        max_date = df_daily.loc[df_daily['post_count'].idxmax(), 'post_date'].strftime('%Y-%m-%d')

        # Add statistics text box
        stats_text = (
            f"Statistics:\n"
            f"Total Posts: {total_posts:,}\n"
            f"Average: {avg_posts:.1f} posts/day\n"
            f"Peak: {max_posts} posts\n"
            f"Peak Date: {max_date}"
        )

        # Add trend analysis
        if len(df_daily) > 1:
            first_week_avg = df_daily['post_count'].head(min(7, len(df_daily))).mean()
            last_week_avg = df_daily['post_count'].tail(min(7, len(df_daily))).mean()
            percent_change = ((last_week_avg - first_week_avg) / first_week_avg) * 100
            
            stats_text += (
                f"\n\nTrend Analysis:\n"
                f"First {min(7, len(df_daily))} days avg: {first_week_avg:.1f}\n"
                f"Last {min(7, len(df_daily))} days avg: {last_week_avg:.1f}\n"
                f"Change: {percent_change:+.1f}%"
            )

        # Calculate hourly patterns
        hourly_avg = df_hourly.groupby('hour')['post_count'].mean()
        peak_hour = hourly_avg.idxmax()
        peak_hour_avg = hourly_avg.max()
        
        stats_text += (
            f"\n\nHourly Patterns:\n"
            f"Peak Hour (UTC): {int(peak_hour):02d}:00\n"
            f"Avg Posts at Peak: {peak_hour_avg:.1f}"
        )

        plt.figtext(
            1.02, 0.6,
            stats_text,
            bbox=dict(
                facecolor='white',
                edgecolor='gray',
                alpha=1.0,
                pad=10
            ),
            fontsize=10,
            verticalalignment='top'
        )

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)

        return fig

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Example usage:
if __name__ == "__main__":
    db_url = 'postgresql://postgres:password@localhost:5433/crawler_db'
    fig = plot_4chan_analysis(db_url, days_back=30)  # Adjust days_back as needed
    if fig:
        plt.show()