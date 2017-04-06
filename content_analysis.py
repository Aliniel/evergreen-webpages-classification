'''FIIT STU, 2017, Data Mining course

Content Analysis module provides a set of functions for natural language
processing. It was made for natural language processing as part of a larger
project in the Data Mining course at FIIT STU, 2017 by Branislav Makan.
'''

import re
# import pandas
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from sklearn.datasets import fetch_20newsgroups


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


def lexical_diversity(bag_of_words):
    '''Get Lexical diversity of the bag of words. The higher the number the more
    diverse the text is - more unique words it uses.
        bag_of_words - Bag or words contained in some text.
    '''
    return len(set(bag_of_words)) / len(bag_of_words)


def get_bag_of_words(text_content):
    '''Get bag of words and legixal diversity for the text_content passed
    as argument.
        text_content - String of the text from which to construct bag of words.
    '''

    # Make a list of words
    bag_of_words = nltk.word_tokenize(text_content)
    lancasetr_stemmer = LancasterStemmer()

    # Remove stopwords, e.g. 'i', 'and', 'on'...
    bag_of_words = [
        lancasetr_stemmer.stem(word)
        for word in bag_of_words
        if word not in stopwords.words('english')
    ]

    return nltk.FreqDist(bag_of_words), lexical_diversity(bag_of_words)


def categorize(train_df, test_df):
    '''Get topic list for the text_content passed as argument. It's recommended
    to normalize the text before using this function.
        text_content - String of the text from which to extract topics.
    '''
    categories = ['alt.atheism', 'soc.religion.christian',
        'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train',
        categories=categories, shuffle=True, random_state=42)

    # del train_df['label']
    # full_df = pandas.concat([train_df, test_df], ignore_index=True)
    # # text_content = pandas.read_json(full_df['page_content'][60], typ='series')

    # all_words = set(
    #     word
    #     for text_content in full_df['page_content']
    #     for word in get_bag_of_words(normalize(text_content))[0].keys()
    #     )
    # print(len(all_words))

    # train_data = full_df.loc[full_df['alchemy_category'] != '?']
    # train_data = train_data.reset_index(drop=True)

    # feature_list = []
    # for i in range(len(train_data)):
    #     page_content = train_data['page_content'][i]
    #     category = train_data['alchemy_category'][i]

    #     page_content = normalize(page_content)
    #     bag_of_words = get_bag_of_words(page_content)[0].keys()
    #     features = {}
    #     for word in all_words:
    #         features[word] = word in bag_of_words
    #     feature_list.append((features, category))

    # print(feature_list)
    import pdb
    pdb.set_trace()

    return train_df, test_df
