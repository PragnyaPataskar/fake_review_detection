# imports
import re
import string
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.sentiment import SentimentIntensityAnalyzer
import textstat
from nrclex import NRCLex

# Download necessary NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def get_review_length(review) -> int:
    """
    Calculate the total number of characters in the review text.
    If the input is not a valid string, return 0.

    Args:
        text: The review text, which can be a string or a non-string type.
    
    Returns:
        int: The length of the review.
    """
    if not isinstance(review, str):
        return 0
    return len(review)


def get_word_count(review) -> int:
    """
    Count the total number of words in the review text.
    If the input is not a valid string, returns 0.

    Args:
        review: The review text.
    
    Returns:
        int: The word count.
    """
    if not isinstance(review, str):
        return 0
    tokens = word_tokenize(review)
    return len(tokens)

def get_avg_word_length(review) -> float:
    """
    Calculate the average word length in the review text.
    If the input is not a valid string, returns 0.

    Args:
        review: The review text.
    
    Returns:
        float: The average length of words.
    """
    if not isinstance(review, str):
        return 0
    tokens = word_tokenize(review)
    if tokens:
        return sum(len(word) for word in tokens) / len(tokens)
    return 0


def get_punctuation_count(review) -> int:
    import string
    if not isinstance(review, str):
        return 0
    return sum(1 for char in review if char in string.punctuation)

def get_exclamation_count(review: str) -> int:
    """
    Count the number of exclamation marks in the text.

    Args:
        review (str): The review text.

    Returns:
        int: The exclamation mark count.
    """
    if not isinstance(review, str):
        return 0
    return review.count('!')

def get_question_count(review: str) -> int:
    """
    Count the number of question marks in the text.

    Args:
        review (str): The review text.

    Returns:
        int: The question mark count.
    """
    if not isinstance(review, str):
        return 0
    return review.count('?')

def get_uppercase_word_count(review) -> int:
    """
    Count the number of fully uppercase words in the review text.
    If the input is not a valid string, returns 0.

    Args:
        review: The review text.
    
    Returns:
        int: The count of uppercase words.
    """
    if not isinstance(review, str):
        return 0
    tokens = word_tokenize(review)
    return sum(1 for word in tokens if word.isupper())

def get_adjectives_adverbs(review: str) -> dict:
    '''
    Count the number of adjectives and adverbs in the text.

    Args:
        text (str): The review text.

    Returns:
        dict: Counts of adjectives and adverbs.
    '''
    pos_tags = pos_tag(word_tokenize(review))
    counts = Counter(tag for word, tag in pos_tags)
    return {'adjectives': counts['JJ'] + counts['JJR'] + counts['JJS'], 'adverbs': counts['RB'] + counts['RBR'] + counts['RBS']}


def get_flesch_kincaid_grade(review: str) -> float:
    '''
    Calculate the Flesch-Kincaid grade level of the review text.

    Args:
        review (str): The review text.

    Returns:
        float: The Flesch-Kincaid grade level.
    '''
    if not isinstance(review, str):
        return 0
    return textstat.flesch_kincaid_grade(review)


def get_sentiment_features(review: str) -> dict:
    """
    Detect sentiment features using NLTK's VADER sentiment analyzer.
    
    This function returns a dictionary containing:
      - neg: Negative sentiment score.
      - neu: Neutral sentiment score.
      - pos: Positive sentiment score.
      - compound: Compound sentiment score.
    
    Args:
        review (str): The review text.
    
    Returns:
        dict: A dictionary with sentiment scores.
    """
    if not isinstance(review, str):
        return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(review)
    return scores

def get_emotional_features(review: str) -> dict:
    """
    Detect emotional features using the NRCLex library.
    
    This function returns a dictionary containing raw emotion counts,
    such as anger, anticipation, disgust, fear, joy, sadness, surprise, and trust.
    
    Args:
        review (str): The review text.
    
    Returns:
        dict: A dictionary with emotion counts. If NRCLex is not installed or
              the input is not valid, returns an empty dictionary.
    """
    if not isinstance(review, str):
        return {}
    
    emotion = NRCLex(review)
    return emotion.raw_emotion_scores # here emotions are returned in dict with each emotion count 