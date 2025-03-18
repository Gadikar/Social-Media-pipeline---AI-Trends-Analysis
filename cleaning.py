from bs4 import BeautifulSoup
import re
import logging


logger = logging.getLogger("4chan client")
logger.propagate = False
logger.setLevel(logging.INFO)

sh = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s"
)
sh.setFormatter(formatter)
logger.addHandler(sh)


def clean_text(comment: str) -> str:
    """
    Clean the comment text by removing HTML tags, entities, and formatting

    Args:
        comment: The raw comment text containing HTML

    Returns:
        str: The cleaned text, or empty string if cleaning fails
    """

    # Early return for invalid input
    if not comment or not isinstance(comment, str):
        logger.warning(f"Invalid comment input: {type(comment)}  : comment {comment}")
        return ""

    try:
        # Convert HTML to plain text
        soup = BeautifulSoup(comment, 'html.parser')

        # Remove quote links
        for quote_link in soup.find_all('a', class_='quotelink'):
            quote_link.decompose()

        text = soup.get_text()

        # HTML entity replacement map
        html_entities = {
            '&gt;': '',
            '&lt;': ' ',
            '&amp;': '&',
            '&#039;': "'",
            '&quot;': '"',
            '>': '',
            '<': ''
        }

        # Replace all HTML entities
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)

        # Remove post references
        text = re.sub(r'>[0-9]+', '', text)

        # Split into lines and clean each line
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            try:
                # Remove any remaining '>' characters at the start of lines
                line = re.sub(r'^>', '', line)
                # Clean extra whitespace within the line
                line = ' '.join(word for word in line.split() if word)
                if line:
                    cleaned_lines.append(line)
            except Exception as e:
                logger.warning(f"Error cleaning line: {e}")
                continue

        # Join lines back together with newlines
        text = '\n'.join(cleaned_lines)
        final_text = text.strip()

        # Return empty string instead of None for consistency
        return final_text if final_text else ""

    except Exception as e:
        logger.error(f"Error in clean_text: {e}")
        return ""
