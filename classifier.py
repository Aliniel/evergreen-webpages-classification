'''
Classifier
==========

FIIT STU, 2017, Data Mining course

Classifier module loads preprocessed data about web pages obtained from kaggle competition
and tries to classify these as evergreen or not evergeen pages.
'''

import pandas
import neurolab
import numpy
import pylab
import pdb
from sklearn.metrics import accuracy_score

from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def load_data(data_type):
    '''Load the preprocessed data.'''

    if data_type == 'train':
        data = pandas.DataFrame(pandas.read_csv('processed_train.tsv', sep='\t'))
    else:
        data = pandas.DataFrame(pandas.read_csv('processed_test.tsv', sep='\t'))

    return data


def run_classifier(fields):
    '''
    Run the classifier for the web page classification.

    fields
    ------

    List of field names to be used for the classification.
    '''

    train_data = load_data('train')

    # Get training and testing sets from the training data
    train_set = train_data.sample(frac=0.8, random_state=200)
    test_set = train_data.drop(train_set.index)

    field_num = len(fields)

    train_set_input = train_set.as_matrix(columns=fields)
    test_set_input = test_set.as_matrix(columns=fields)

    train_set_class = train_set.as_matrix(columns=['is_evergreen'])
    test_set_class = test_set.as_matrix(columns=['is_evergreen'])

    # Create multilayered feed-forward neural network
    ranges = field_num * [[0, 1]]  # Each range is also an input neuron
    topology = [10, 1]  # Hidden + output layer
    net = neurolab.net.newff(ranges, topology)
    net.trainf = neurolab.train.train_rprop
    net.errorf = neurolab.error.MSE()

    # Train the network
    error = net.train(
        train_set_input,
        train_set_class,
        epochs=500,
        show=10,
        goal=0.02,
        lr=0.07,
        adapt=False,  # Strange behavior when set to True
        rate_dec=0.5,
        rate_inc=1.2,
        rate_min=1e-9,
        rate_max=50,
    )

    # Simulate the network
    out = net.sim(test_set_input)
    out = out.round().astype(int)
    print("Classification accuracy: %s." % (accuracy_score(out, test_set_class)))

    # Plot result
    pylab.subplot(211)
    pylab.plot(error)
    pylab.xlabel('Epoch number')
    pylab.ylabel('error (default SSE)')

    pylab.show()

    # Text only clssification
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

    text_classifier = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    text_classifier.fit(
        train_set['textcontent'],
        train_set['is_evergreen'],
    )

    predicted = text_classifier.predict(test_set['textcontent'])
    original = test_set['is_evergreen'].as_matrix()

    print("Text classification accuracy: %s" % (accuracy_score(predicted, original)))


FIELDS = [
    'common_link_ratio_3',  # 0.11
    'compression_ratio',  # 0.06
    'iframe_ratio',  # 0.19
    'html_ratio',  # 0.05
    'image_ratio',  # 0.04
    'hyperlink_text_ratio',  # -0.17
    'raw_text_character_count',  # 0.1
    'link_count',  # 0.08
    'spelling_errors_ratio',  # -0.06
]
run_classifier(FIELDS)
