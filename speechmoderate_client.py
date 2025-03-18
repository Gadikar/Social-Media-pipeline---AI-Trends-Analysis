import os
from dotenv import load_dotenv
import requests
import logging

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("moderate_hatespeech_client")
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
logger.addHandler(sh)

class ModerateHateSpeechClient:
    BASE_URL = "https://api.moderatehatespeech.com/api/v1"
    CONF_THRESHOLD = 0.9

    def __init__(self):
        self.api_key = os.getenv('MODERATE_HATESPEECH_API_KEY')
        if not self.api_key:
            raise ValueError("MODERATE_HATESPEECH_API_KEY environment variable is required")

    def check_comment(self, comment):
        """
        Check if a comment contains hate speech.
        
        Args:
            comment (str): The text to check for hate speech
            
        Returns:
            bool: True if hate speech is detected with confidence above threshold, False otherwise
        """
        try:
            data = {
                "token": self.api_key,
                "text": comment
            }
            
            response = requests.post(
                f"{self.BASE_URL}/moderate/", 
                json=data
            )
            response.raise_for_status()
            result = response.json()
            
            logger.debug(f"API Response: {result}")
            
            if result["class"] == "flag" and float(result["confidence"]) > self.CONF_THRESHOLD:
                logger.info(f"Hate speech detected in comment with confidence {result['confidence']}")
                 
                
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling hate speech API: {str(e)}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing API response: {str(e)}")
            raise

'''
if __name__ == "__main__":
    client = ModerateHateSpeechClient()
    test_text = ""
    result = client.check_comment(test_text)
    print(f"Hate speech detected: {result}")
    '''