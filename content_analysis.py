'''FIIT STU, 2017, Data Mining course

Content Analysis module provides a set of functions for natural language
processing. It was made for natural language processing as part of a larger
project in the Data Mining course at FIIT STU, 2017 by Branislav Makan.
'''

import nltk
import re
from nltk.corpus import stopwords


def normalize(text_content):
    '''Normalize the natural text passed as argument by removing all non-letter
    characters and transforing the characters to lowercase. 
        text_content - String of the text to be normalized.
    '''

    # Transform the text to lowercase
    text_content = text_content.lower()

    # Remove all non-letters characters
    regex = re.compile('[^a-z]')
    text_content = regex.sub(' ', text_content)

    return text_content


def get_bag_of_words(text_content):
    '''Get bag of words for the text_content passed as argument.
        text_content - String of the text from which to construct bag of words.
    '''

    # Make a list of words
    bag_of_words = text_content.split()

    # Remove stopwords, e.g. 'i', 'and', 'on'...
    bag_of_words = [
        word for word in bag_of_words
        if word not in stopwords.words('english')
    ]

    return bag_of_words


def get_topics(text_content):
    '''Get topic list for the text_content passed as argument. It's recommended
    to normalize the text before using this function.
        text_content - String of the text from which to extract topics.
    '''
    # TODO implement model for retrieving topics
