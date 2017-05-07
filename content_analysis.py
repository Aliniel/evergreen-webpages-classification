'''FIIT STU, 2017, Data Mining course

Content Analysis module provides a set of functions for natural language
processing. It was made for natural language processing as part of a larger
project in the Data Mining course at FIIT STU, 2017 by Branislav Makan.
'''

import re
import pandas
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier


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


def categorize(train_df):
    '''Get topic list for the text_content passed as argument. It's recommended
    to normalize the text before using this function.
        text_content - String of the text from which to extract topics.
    '''

    # Parse text content from data frame into a raw text list
    alchemy_labelled_df = train_df.loc[train_df['alchemy_category'] != '?']
    labelled_text_content = []
    for page_content in alchemy_labelled_df['page_content']:
        json_content = pandas.read_json(page_content, typ='Series')

        if json_content.body is not None:
            labelled_text_content.append(json_content.body)
        else:
            labelled_text_content.append(json_content.title)

        if labelled_text_content[-1] is None:
            labelled_text_content[-1] = ''

    alchemy_unlabelled_df = train_df.loc[train_df['alchemy_category'] == '?']
    unlabelled_text_content = []
    for page_content in alchemy_unlabelled_df['page_content']:
        json_content = pandas.read_json(page_content, typ='Series')

        if json_content.body is not None:
            unlabelled_text_content.append(json_content.body)
        else:
            unlabelled_text_content.append(json_content.title)

        if unlabelled_text_content[-1] is None:
            unlabelled_text_content[-1] = ''

    # Build the classifier pipeline: build basg of words,
    # then get frequencies and train Naive Bayes Model
    # text_classifier = Pipeline([
    #     ('vect', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', MultinomialNB()),
    # ])

    # SVG Classifier
    text_classifier = Pipeline([
        ('vect', CountVectorizer(
            ngram_range=(1, 2)
        )),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(
            loss='hinge',
            penalty='l2',
            alpha=1e-3,
            n_iter=5,
            random_state=42)),
    ])

    # --- This part was used to find the optimal parameters for the model ---

    # parameters = {
    #     'vect__ngram_range': [(1, 1), (1, 2)],
    #     'tfidf__use_idf': (True, False),
    #     'clf__alpha': (1e-2, 1e-3),
    # }

    # gs_clf = GridSearchCV(
    #     text_classifier,
    #     parameters,
    #     n_jobs=-1
    # )
    # gs_clf = gs_clf.fit(
    #     labelled_text_content[:-2000],
    #     alchemy_labelled_df['alchemy_category'][:-2000]
    # )
    # for param_name in sorted(parameters.keys()):
    #     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

    text_classifier = text_classifier.fit(
        labelled_text_content,
        alchemy_labelled_df['alchemy_category']
    )

    predicted = text_classifier.predict(unlabelled_text_content)
    train_df.loc[train_df['alchemy_category'] == '?', ('alchemy_category')] = predicted

    return train_df
