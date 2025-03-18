from datetime import datetime, timedelta

import streamlit as st
from dotenv import load_dotenv

from utils import get_date_range, load_reddit_data_chunked, load_4chan_data_chunked, analyze_hourly_patterns, \
    analyze_daily_patterns, display_comprehensive_engagement_summary, display_hourly_patterns, display_daily_patterns, \
    get_top_ai_users_reddit, display_ai_engagement_analysis, get_network_data, display_network_analysis

# Load environment variables and set up configuration
load_dotenv()

from rag_system import (
    initialize_vector_store, generate_response, get_topic_specific_questions,
    generate_query_focused_visualization
)
# Set page config
st.set_page_config(
    page_title="Platform Engagement Analysis",
    page_icon="📊",
    layout="wide"
)


# Import utility functions from previous files
# [Keep all the utility functions from the original files:
# get_cache_key, save_cache, load_cache, init_connection, get_date_range,
# and all data loading and processing functions]

def main():
    st.title("Platform Engagement Analysis")

    try:
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Hourly/Daily Pattern",
            "AI Engagement",
            "Network Analysis",
            "RAG Analysis"
        ])

        # Platform selector in sidebar (available for all tabs)
        platform = st.sidebar.radio("Select Platform", ["Reddit", "4chan"])

        # Get date range for selected platform
        min_date, max_date = get_date_range(platform.lower())
        min_dt = datetime.fromtimestamp(min_date)
        max_dt = datetime.fromtimestamp(max_date)

        # Date range selector in sidebar (available for all tabs)
        date_range = st.sidebar.date_input(
            "Select Date Range (Max 30 days)",
            value=(max_dt - timedelta(days=7), max_dt),
            min_value=min_dt.date(),
            max_value=max_dt.date()
        )

        if len(date_range) == 2:
            start_date, end_date = date_range

            # Ensure the date range isn't too large
            if (end_date - start_date).days > 30:
                st.error("Please select a date range of 30 days or less.")
                return

            # Convert dates to timestamps
            start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
            end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

            # Tab 1: Hourly/Daily Pattern
            with tab1:
                with st.spinner(f"Loading {platform} data..."):
                    if platform == "Reddit":
                        df = load_reddit_data_chunked(start_timestamp, end_timestamp)
                    else:
                        df = load_4chan_data_chunked(start_timestamp, end_timestamp)

                    if not df.empty:
                        hourly_stats = analyze_hourly_patterns(df, platform)
                        daily_stats = analyze_daily_patterns(df, platform)
                        display_comprehensive_engagement_summary(hourly_stats, daily_stats, platform)

                        # Analysis type selector within tab
                        analysis_type = st.selectbox(
                            "Select Analysis Type",
                            ["Hourly Patterns", "Daily Patterns"]
                        )

                        if analysis_type == "Hourly Patterns":
                            st.header(f"Hourly Engagement Patterns - {platform}")
                            display_hourly_patterns(hourly_stats, platform)
                        else:
                            st.header(f"Daily Engagement Patterns - {platform}")
                            display_daily_patterns(daily_stats, platform)
                    else:
                        st.warning("No data found for the selected date range.")

            # Tab 2: AI Engagement
            with tab2:
                if platform == "Reddit":
                    with st.spinner("Analyzing AI discussions..."):
                        ai_data = get_top_ai_users_reddit(start_timestamp, end_timestamp)
                        if not ai_data.empty:
                            display_ai_engagement_analysis(ai_data)
                        else:
                            st.warning("No AI-related discussions found in the selected date range.")
                else:
                    st.warning("AI engagement analysis is currently only available for Reddit data.")

            # Tab 3: Network Analysis
            with tab3:
                if platform == "Reddit":
                    with st.spinner("Analyzing network data..."):
                        network_df = get_network_data(start_timestamp, end_timestamp)
                        if not network_df.empty:
                            display_network_analysis(network_df)
                        else:
                            st.warning("No network data found for the selected date range.")
                else:
                    st.warning("Network analysis is currently only available for Reddit data.")

            with tab4:
                render_rag_tab()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your settings and try again.")


def render_rag_tab():
    """Render the RAG Analysis System tab content"""
    # st.title("Social Media Analysis RAG System")
    # st.markdown("---")

    # Create two columns for the main layout
    left_col, right_col = st.columns([6, 4])

    # Sidebar settings for RAG
    with st.sidebar:
        topic = st.selectbox("Select Topic", ["Politics", "AI"], key="rag_topic")
        data_source = st.selectbox(
            "Select Data Source",
            ["reddit_posts", "reddit_comments", "chan_posts"],
            key="rag_source"
        )

    # Left column: Q&A Section
    with left_col:
        st.markdown("### Query Configuration")
        with st.container():
            questions = get_topic_specific_questions(topic, data_source)
            selected_question = st.selectbox("Select your question:", questions, key="rag_question")
            custom_question = st.text_input("Or enter your custom question:", key="rag_custom")
            analyze_button = st.button("Generate Analysis", use_container_width=True, key="rag_analyze")

        if analyze_button:
            with st.spinner(f"Analyzing {topic} data..."):
                collection = initialize_vector_store(data_source, topic)
                
                if collection:
                    question = custom_question if custom_question else selected_question
                    results = collection.query(
                        query_texts=[question],
                        n_results=5,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    if results['documents'][0]:
                        context = "\n\n".join(results['documents'][0])
                        response = generate_response(context, question, topic)
                        
                        st.markdown("### Analysis Results")
                        st.markdown(response)
                        
                        # Right column: Visualization
                        with right_col:
                            st.markdown("### Data Visualization")
                            try:
                                html_path = generate_query_focused_visualization(results, question, topic)
                                if html_path:
                                    st.components.v1.html(
                                        open(html_path, 'r', encoding='utf-8').read(),
                                        height=600
                                    )
                            except Exception as e:
                                st.error(f"Visualization error: {str(e)}")

                        # Source Documents
                        st.markdown("### Source Documents")
                        with st.expander("View Source Documents", expanded=False):
                            for idx, (doc, dist, meta) in enumerate(zip(
                                results['documents'][0],
                                results['distances'][0],
                                results['metadatas'][0]
                            ), 1):
                                with st.container():
                                    st.markdown(f"**Document {idx}**")
                                    st.markdown(f"Relevance Score: {(1 - dist):.2f}")
                                    st.json(meta)
                                    st.markdown(f"```\n{doc[:200]}...\n```")
                                    st.markdown("---")
                    else:
                        st.error(f"No relevant {topic} documents found!")
                else:
                    st.error("Failed to initialize vector store")

if __name__ == "__main__":
    main()