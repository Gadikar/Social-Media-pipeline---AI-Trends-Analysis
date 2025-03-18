import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import nltk
import string
import re
import colorsys

def get_ai_keywords():
    """
    Get list of AI-related keywords
    """
    ai_keywords = {
        'artificial', 'intelligence', 'machine', 'learning', 'deep', 'neural', 
        'networks', 'natural', 'language', 'processing', 'reinforcement', 'ethics', 
        'alignment', 'chatgpt', 'gpt', 'bert', 'llama', 'stable', 'diffusion', 
        'midjourney', 'openai', 'google', 'deepmind', 'anthropic', 'source', 
        'safety', 'regulation', 'prompt', 'engineering', 'fine-tuning', 
        'transformers', 'gans', 'automl', 'large', 'models', 'tensorflow', 
        'pytorch', 'scikit-learn', 'hugging', 'face', 'langchain', 'onnx', 
        'cuda', 'acceleration', 'bias', 'privacy', 'concerns', 'surveillance', 
        'autonomous', 'weapons', 'deepfake', 'technology', 'gaming', 'healthcare', 
        'finance', 'generative', 'applications', 'agi', 'skynet', 'paperclip', 
        'maximizer', 'doom', 'supremacy', 'stochastic', 'parrot', 'sparks', 
        'overlords'
    }
    additional_terms = {
        'ai', 'ml', 'dl', 'nlp', 'cv', 'transformer',
        'dataset', 'inference', 'supervised', 'unsupervised', 'optimization'
    }
    return ai_keywords.union(additional_terms)

def extract_ai_terms(text, ai_keywords):
    """
    Extract individual AI-related terms from text
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = text.split()
    ai_terms = [word for word in words if word in ai_keywords]
    return ai_terms

def process_text_chunks(csv_path, ai_keywords, chunk_size=10000):
    """
    Process CSV file in chunks and count unique AI terms
    """
    term_counts = Counter()
    total_records = 0
    
    chunks = pd.read_csv(csv_path, chunksize=chunk_size)
    
    print("Processing CSV file in chunks...")
    for chunk in tqdm(chunks):
        valid_texts = chunk[
            (chunk['cleaned_text'].notna()) & 
            (chunk['cleaned_text'] != '')
        ]['cleaned_text']
        
        for text in valid_texts:
            terms = extract_ai_terms(str(text), ai_keywords)
            if terms:
                term_counts.update(terms)
                total_records += 1
    
    return term_counts, total_records

def get_color_palette():
    """
    Generate a vibrant color palette
    """
    # Define base colors (in HSV)
    colors = [
        (0.7, 0.8, 0.95),    # Blue
        (0.35, 0.8, 0.95),   # Green
        (0.05, 0.8, 0.95),   # Orange
        (0.55, 0.8, 0.95),   # Purple
        (0.15, 0.8, 0.95),   # Yellow-Orange
        (0.85, 0.8, 0.95),   # Pink
        (0.45, 0.8, 0.95),   # Teal
        (0.95, 0.8, 0.95),   # Red
    ]
    return colors

def generate_unique_ai_wordcloud(csv_path, chunk_size=10000):
    """
    Generate a professional, colorful word cloud showing unique AI terms
    """
    plt.style.use('default')
    
    ai_keywords = get_ai_keywords()
    term_counts, total_records = process_text_chunks(csv_path, ai_keywords, chunk_size)
    
    print(f"\nFound AI-related terms in {total_records:,} records")
    
    # Get color palette
    colors = get_color_palette()
    
    def multi_color_func(word=None, font_size=None, position=None, orientation=None, **kwargs):
        """
        Generate different colors based on word frequency and position
        """
        # Use word's font size to determine color intensity
        random_state = np.random.RandomState(font_size)
        color_index = random_state.randint(0, len(colors))
        base_h, base_s, base_v = colors[color_index]
        
        # Vary the saturation and value based on font size
        s = base_s * (0.7 + 0.3 * (font_size / 120))  # 120 is max_font_size
        v = base_v * (0.7 + 0.3 * (font_size / 120))
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(base_h, s, v)
        return f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"
    
    # Configure word cloud
    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        max_words=150,
        prefer_horizontal=0.6,
        relative_scaling=0.3,
        min_font_size=8,
        max_font_size=120,
        random_state=42,
        collocations=False,
        color_func=multi_color_func,
        margin=2,
        scale=3,
        repeat=False
    ).generate_from_frequencies(term_counts)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    plt.imshow(wordcloud, interpolation='lanczos')
    plt.axis('off')
    
    # Add title with shadow effect
    plt.text(0.5, 0.95, 'AI-Related Terms Frequency Analysis', 
            horizontalalignment='center',
            transform=fig.transFigure,
            fontsize=16,
            fontweight='bold',
            color='#2c3e50')  # Dark blue-gray
    
    plt.text(0.5, 0.92, f'Analysis of {total_records:,} posts',
            horizontalalignment='center',
            transform=fig.transFigure,
            fontsize=10,
            color='#7f8c8d')  # Medium gray
    
    # Save with high quality
    output_filename = 'ai_wordcloud_colorful.png'
    plt.savefig(output_filename, 
                bbox_inches='tight', 
                dpi=300, 
                facecolor='white', 
                edgecolor='none',
                pad_inches=0.2)
    
    print(f"\nWord cloud saved as {output_filename}")
    
    # Print statistics
    print("\nTop 20 most frequent AI terms:")
    for term, count in term_counts.most_common(20):
        print(f"{term}: {count:,}")
    
    print(f"\nPercentage of posts with AI terms: {(total_records/chunk_size)*100:.2f}%")

if __name__ == "__main__":
    CHUNK_SIZE = 10000
    generate_unique_ai_wordcloud(
        csv_path='chan_posts1.csv',
        chunk_size=CHUNK_SIZE
    )