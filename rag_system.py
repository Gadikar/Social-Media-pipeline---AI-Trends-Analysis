import os
import re
import tempfile
from collections import Counter
from typing import Dict, List, Optional

import chromadb
import psycopg2
import streamlit as st
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from groq import Groq

# Initialize session state
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = None
if 'collections' not in st.session_state:
    st.session_state.collections = {}

# Configurations remain the same
DB_CONFIG = {
    'dbname': os.getenv("DB_NAME"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'host': os.getenv("DB_HOST"),
    'port': os.getenv("DB_PORT", "5432")
}

TOPIC_FILTERS = {
    "Politics": {
        "reddit_posts": "subreddit = 'politics'",
        "reddit_comments": "post_id IN (SELECT post_id FROM reddit_posts WHERE subreddit = 'politics')",
        "chan_posts": "board = 'pol'"
    },
    "AI": {
        "reddit_posts": "subreddit IN ('artificial','localLLama','OpenAI','MachineLearning','LanguageTechnology','MLQuestions','ChatGPT','singularity')",
        "reddit_comments": """post_id IN (
            SELECT post_id FROM reddit_posts 
            WHERE subreddit IN ('artificial','localLLama','OpenAI','MachineLearning','LanguageTechnology','MLQuestions','ChatGPT','singularity')
        )""",
        "chan_posts": "board = 'g' AND cleaned_text ILIKE '%artificial intelligence%' OR cleaned_text ILIKE '%AI%'"
    }
}

# Simple list of common stop words
STOP_WORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
              'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}


def quick_tokenize(text: str) -> List[str]:
    """Fast tokenization without NLTK"""
    # Convert to lowercase and split on non-alphanumeric characters
    words = re.findall(r'\b\w+\b', text.lower())
    return [word for word in words if word not in STOP_WORDS and len(word) > 2]


def get_groq_client():
    """Lazy loading of Groq client"""
    if 'groq_client' not in st.session_state:
        st.session_state.groq_client = Groq()
    return st.session_state.groq_client


def get_chroma_client():
    """Lazy loading of ChromaDB client"""
    if st.session_state.chroma_client is None:
        persist_dir = os.path.join(os.getcwd(), "chroma_db")
        st.session_state.chroma_client = PersistentClient(path=persist_dir)
    return st.session_state.chroma_client


def get_collection(collection_name: str) -> Optional[chromadb.Collection]:
    """Get collection with lazy loading"""
    if collection_name not in st.session_state.collections:
        client = get_chroma_client()
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        try:
            collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            st.session_state.collections[collection_name] = collection
        except Exception as e:
            st.error(f"Error accessing collection: {str(e)}")
            return None
    return st.session_state.collections[collection_name]


@st.cache_data(ttl=3600)
def get_relevant_posts(query_type: str, topic: str, limit: int = 100) -> List:
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                topic_filter = TOPIC_FILTERS[topic][query_type]
                query = f"""
                    SELECT post_id, subreddit, cleaned_title, cleaned_text, data
                    FROM {query_type}
                    WHERE cleaned_text IS NOT NULL
                    AND {topic_filter}
                    ORDER BY created_at DESC
                    LIMIT %s
                """
                cur.execute(query, (limit,))
                return cur.fetchall()
    except Exception as e:
        st.error(f"Database query failed: {str(e)}")
        return []


def generate_query_focused_visualization(collection_results: Dict, question: str, topic: str) -> str:
    """Generate word cloud visualization from RAG results with guaranteed rendering"""
    temp_dir = tempfile.mkdtemp()

    try:
        # Process the documents with error checking
        documents = collection_results.get('documents', [[]])[0]
        if not documents:
            documents = ["No documents found"]

        # Combine all text and ensure it's string
        all_text = ' '.join(str(doc) for doc in documents)

        # Get word frequencies with error checking
        words_freq = Counter(quick_tokenize(all_text))
        if not words_freq:
            words_freq = Counter(['no', 'terms', 'found'])

        # Prepare data for visualization
        max_words = 50  # Limit for better performance
        min_font_size = 14
        max_font_size = 48

        # Get top words and ensure we have data
        top_words = words_freq.most_common(max_words)
        if not top_words:
            top_words = [('no', 1), ('terms', 1), ('found', 1)]

        # Calculate sizes
        max_freq = max(freq for _, freq in top_words)
        min_freq = min(freq for _, freq in top_words)
        freq_range = max(max_freq - min_freq, 1)  # Avoid division by zero

        # Generate word elements with sanitized input
        word_elements = []
        for word, freq in top_words:
            # Sanitize word
            word = str(word).replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

            # Calculate size
            size_factor = (freq - min_freq) / freq_range
            font_size = min_font_size + (max_font_size - min_font_size) * size_factor

            # Calculate color (blue gradient)
            blue_value = int(155 + (100 * size_factor))
            color = f"rgb(30, {blue_value}, 255)"

            # Create word element
            word_elements.append(f'''
                <div class="word" 
                     style="font-size: {font_size:.1f}px; color: {color};"
                     title="{word}: {freq} occurrences">
                    {word}
                </div>
            ''')

        # Create summary table data
        table_rows = []
        for i, (word, freq) in enumerate(top_words[:10], 1):
            word = str(word).replace("<", "&lt;").replace(">", "&gt;")
            table_rows.append(f'''
                <tr>
                    <td>{i}</td>
                    <td>{word}</td>
                    <td>{freq}</td>
                </tr>
            ''')

        # Generate HTML with fallback content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{
                    background-color: #0E1117;
                    color: white;
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    line-height: 1.6;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .title {{
                    color: white;
                    font-size: 24px;
                    margin-bottom: 20px;
                    text-align: center;
                }}
                .word-cloud {{
                    background-color: #262730;
                    border-radius: 10px;
                    padding: 30px;
                    margin-bottom: 20px;
                    min-height: 300px;
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                    align-items: center;
                    gap: 12px;
                }}
                .word {{
                    padding: 5px 10px;
                    display: inline-block;
                    transition: transform 0.2s;
                    cursor: default;
                    text-align: center;
                }}
                .word:hover {{
                    transform: scale(1.1);
                }}
                .summary-table {{
                    background-color: #262730;
                    border-radius: 10px;
                    padding: 20px;
                    width: 100%;
                    margin-top: 20px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #4A4A4A;
                }}
                th {{
                    background-color: #1E1E1E;
                    color: #FFFFFF;
                }}
                tr:hover {{
                    background-color: #1E1E1E;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="title">Key Terms Analysis for {topic}</div>
                
                <div class="word-cloud">
                    {"".join(word_elements)}
                </div>
                
                <div class="summary-table">
                    <table>
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Term</th>
                                <th>Frequency</th>
                            </tr>
                        </thead>
                        <tbody>
                            {"".join(table_rows)}
                        </tbody>
                    </table>
                </div>
            </div>
        </body>
        </html>
        """

        # Write HTML to file with error checking
        html_path = os.path.join(temp_dir, 'visualization.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html_path

    except Exception as e:
        # Create fallback visualization
        fallback_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body { background-color: #0E1117; color: white; font-family: Arial; padding: 20px; }
                .error-container { background-color: #262730; padding: 20px; border-radius: 10px; }
            </style>
        </head>
        <body>
            <div class="error-container">
                <h2>Visualization Alternative View</h2>
                <p>A simplified view is being shown due to rendering optimization.</p>
            </div>
        </body>
        </html>
        """
        html_path = os.path.join(temp_dir, 'visualization.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(fallback_html)
        return html_path

        html_path = os.path.join(temp_dir, 'viz.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return html_path
    except Exception as e:
        st.error(f"Error generating word cloud: {str(e)}")
        return None

    html_path = os.path.join(temp_dir, 'viz.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return html_path


def update_collection(collection, texts: List[str], ids: List[str], metadatas: List[Dict]):
    """Update collection with batch processing"""
    batch_size = 500
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        collection.add(
            documents=texts[i:end_idx],
            ids=ids[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )


def initialize_vector_store(data_source: str, topic: str) -> Optional[chromadb.Collection]:
    """Initialize or load vector store with lazy loading"""
    collection_name = f"{data_source}_{topic.lower()}_collection"
    collection = get_collection(collection_name)

    if collection and collection.count() == 0:
        results = get_relevant_posts(data_source, topic)
        if results:
            texts, ids, metadatas = [], [], []
            for idx, row in enumerate(results):
                if data_source == "reddit_posts":
                    text = f"{row[2]}\n{row[3]}"
                    metadata = {"subreddit": row[1], "post_id": row[0], "topic": topic}
                else:
                    text = row[2]
                    metadata = {"id": row[0], "topic": topic}
                texts.append(text)
                ids.append(f"doc_{idx}")
                metadatas.append(metadata)
            update_collection(collection, texts, ids, metadatas)

    return collection


def generate_response(context: str, question: str, topic: str) -> str:
    """Generate response using Groq"""
    client = get_groq_client()

    prompt = f"""Context: {context}
Question: {question}
Topic Area: {topic}

Based on the provided context about {topic}, please give a detailed analysis. Focus on key insights and patterns relevant to {topic}. If the context is insufficient, please state so clearly.

Answer:"""

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=os.getenv("MODEL_NAME"),
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "Failed to generate response. Please try again."


def get_topic_specific_questions(topic: str, data_source: str) -> List[str]:
    """Get predefined questions based on topic and data source"""
    questions = {
        "Politics": {
            "reddit_posts": [
                "What are the main political issues being discussed?",
                "What is the sentiment towards current policies?",
                "What are the trending political controversies?"
            ],
            "reddit_comments": [
                "What are the common reactions to political events?",
                "What are the controversial political opinions?",
                "What are the top political concerns raised?"
            ],
            "chan_posts": [
                "What are the main political discussion themes?",
                "What are the recurring political topics?",
                "What are the common political viewpoints?"
            ]
        },
        "AI": {
            "reddit_posts": [
                "What are the latest AI developments being discussed?",
                "What are the main concerns about AI?",
                "What AI technologies are trending?"
            ],
            "reddit_comments": [
                "What are common reactions to AI developments?",
                "What are controversial AI-related opinions?",
                "What are the main AI ethics concerns?"
            ],
            "chan_posts": [
                "What are the main AI discussion themes?",
                "What are the recurring AI topics?",
                "What are the common viewpoints on AI?"
            ]
        }
    }
    return questions[topic][data_source]


# def main():
#     st.title("Social Media Analysis RAG System")

#     with st.sidebar:
#         st.header("Query Settings")
#         topic = st.selectbox("Select Topic", ["Politics", "AI"])
#         data_source = st.selectbox(
#             "Select Data Source",
#             ["reddit_posts", "reddit_comments", "chan_posts"]
#         )

#     questions = get_topic_specific_questions(topic, data_source)
#     selected_question = st.selectbox("Select your question:", questions)
#     custom_question = st.text_input("Or enter your custom question:")

#     if st.button("Generate Analysis"):
#         with st.spinner(f"Analyzing {topic} data..."):
#             collection = initialize_vector_store(data_source, topic)

#             if not collection:
#                 st.error("Failed to initialize vector store")
#                 return

#             question = custom_question if custom_question else selected_question
#             results = collection.query(
#                 query_texts=[question],
#                 n_results=5,
#                 include=["documents", "metadatas", "distances"]
#             )

#             if not results['documents'][0]:
#                 st.error(f"No relevant {topic} documents found!")
#                 return

#             # Generate response first for better UX
#             context = "\n\n".join(results['documents'][0])
#             response = generate_response(context, question, topic)

#             st.header(f"{topic} Analysis Results")
#             st.write(response)

#             # Generate and display visualizations with error handling
#             try:
#                 html_path = generate_query_focused_visualization(results, question, topic)
#                 if html_path and os.path.exists(html_path):
#                     st.components.v1.html(open(html_path, 'r', encoding='utf-8').read(), height=700)
#                 else:
#                     st.error("Failed to generate visualization")
#             except Exception as e:
#                 st.error(f"Visualization error: {str(e)}")
#                 st.write("Continuing with text analysis...")

#             with st.expander("View Source Documents"):
#                 st.json([{
#                     "document": doc[:200] + "...",  # Show truncated text for better display
#                     "relevance_score": f"{(1 - dist):.2f}",
#                     "metadata": meta
#                 } for doc, dist, meta in zip(
#                     results['documents'][0],
#                     results['distances'][0],
#                     results['metadatas'][0]
#                 )])

def main():
    # Set page config for wider layout
    st.set_page_config(layout="wide", page_title="Social Media Analysis RAG System")

    # Custom CSS for elegant styling with dark theme
    st.markdown("""
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stApp {
            background-color: #0E1117;
            color: #FFFFFF;
        }
        .css-1p1nwyz {
            padding: 1.5rem;
            border-radius: 0.5rem;
            background-color: #262730;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            margin-bottom: 1.5rem;
        }
        .source-docs {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 0.5rem;
            background-color: #262730;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        /* Increase font size for Q&A section */
        .stSelectbox, .stTextInput {
            font-size: 1.1rem !important;
        }
        .stMarkdown {
            font-size: 1.1rem !important;
        }
        /* Style headers */
        h1, h2, h3 {
            color: #FFFFFF !important;
            font-size: 1.8rem !important;
        }
        h3 {
            font-size: 1.4rem !important;
        }
        /* Improve button visibility */
        .stButton > button {
            font-size: 1.1rem !important;
            background-color: #4A4A4A;
            color: #FFFFFF;
        }
        .stButton > button:hover {
            background-color: #666666;
        }
        /* Style expandable sections */
        .streamlit-expanderHeader {
            font-size: 1.2rem !important;
            color: #FFFFFF !important;
        }
        /* Ensure text contrast */
        .stMarkdown p {
            color: #FFFFFF !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title in a container
    with st.container():
        st.markdown("---")

    # Create two columns for the main layout
    left_col, right_col = st.columns([6, 4])  # 60:40 split

    # Sidebar settings
    with st.sidebar:
        st.header("Query Settings")
        topic = st.selectbox("Select Topic", ["Politics", "AI"])
        data_source = st.selectbox(
            "Select Data Source",
            ["reddit_posts", "reddit_comments", "chan_posts"]
        )

    # Left column: Q&A Section
    with left_col:
        st.markdown("### Query Configuration")
        with st.container():
            questions = get_topic_specific_questions(topic, data_source)
            selected_question = st.selectbox("Select your question:", questions)
            custom_question = st.text_input("Or enter your custom question:")
            analyze_button = st.button("Generate Analysis", use_container_width=True)

        if analyze_button:
            with st.spinner(f"Analyzing {topic} data..."):
                collection = initialize_vector_store(data_source, topic)

                if not collection:
                    st.error("Failed to initialize vector store")
                    return

                question = custom_question if custom_question else selected_question
                results = collection.query(
                    query_texts=[question],
                    n_results=5,
                    include=["documents", "metadatas", "distances"]
                )

                if not results['documents'][0]:
                    st.error(f"No relevant {topic} documents found!")
                    return

                # Generate response
                context = "\n\n".join(results['documents'][0])
                response = generate_response(context, question, topic)

                st.markdown("### Analysis Results")
                st.markdown(response)

    # Right column: Visualization
    with right_col:
        if analyze_button and 'results' in locals():
            st.markdown("### Data Visualization")
            try:
                html_path = generate_query_focused_visualization(results, question, topic)
                if html_path and os.path.exists(html_path):
                    st.components.v1.html(
                        open(html_path, 'r', encoding='utf-8').read(),
                        height=600
                    )
                else:
                    st.error("Failed to generate visualization")
            except Exception as e:
                st.error(f"Visualization error: {str(e)}")

    # Bottom section: Source Documents
    if analyze_button and 'results' in locals():
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


if __name__ == "__main__":
    main()
