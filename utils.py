import os
import logging
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


def get_project_dir(sub_dir: str) -> str:
    """Return path to a project subdirectory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), sub_dir))


def configure_logging() -> None:
    """Configures logging"""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def download_nltk_resources():
    """Download required NLTK resources."""
    resources = ['punkt', 'stopwords', 'punkt_tab', 'wordnet']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logging.info(f"Downloaded NLTK resource: {resource}")
        except Exception as e:
            logging.warning(f"Could not download {resource}: {e}")


def preprocess_text(text, keep_negations):
    """
    Preprocess a single text string.

    Steps:
    1. Convert to lowercase
    2. Remove punctuation (except apostrophes)
    3. Tokenize
    4. Remove stopwords (keeping negations)
    5. Lemmatize
    6. Join tokens back to string

    Args:
        text (str): Input text
        keep_negations (list): List of negation words to keep

    Returns:
        str: Preprocessed text
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation (keep apostrophes)
    punctuation = string.punctuation.replace("'", "")
    text = text.translate(str.maketrans('', '', punctuation))

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Remove stopwords (keep negations)
    stop_words = set(stopwords.words('english'))
    negations = set(keep_negations)
    stop_words = stop_words - negations

    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]

    # 5. Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    # 6. Join tokens
    return ' '.join(tokens)