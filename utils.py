# Miscellaneous utils functions
from unidecode import unidecode
import re

def clean_text(text):
    """
    Cleans the input text by performing the following operations:
    1. Adds a space after periods if the next character is a capital letter.
    2. Converts the text to its closest ASCII representation.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    text = add_space_after_period(text)
    text = unidecode(text)
    return text

def add_space_after_period(text):
    """
    Adds a space after periods if the next character is a capital letter.
    This helps in maintaining proper spacing in sentences where periods are followed
    directly by capital letters without a space.

    Args:
        text (str): The input text to be corrected.

    Returns:
        str: The corrected text with spaces added after periods where necessary.
    """
    pattern = r'\.([A-Z])'
    replacement = r'. \1'
    corrected_text = re.sub(pattern, replacement, text)
    return corrected_text

    # Example usage
    # text = "This is a test. This should be detected.A"
    # corrected_text = add_space_after_period(text)
    # print(corrected_text)
