import pandas as pd
import numpy as np
import regex as re
import datetime
from datetime import datetime
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import h2o
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
import pickle
from sklearn.model_selection import train_test_split, cross_val_score

import warnings


def create_w2v_vectors(list_of_words, loaded_model, vec_size=300):
    """
    Returns the average vector for a list of words using a given Word2Vec model.
    """
    word_vectors = [loaded_model.wv[word] if loaded_model.wv.has_index_for(word) else 
                    np.zeros(vec_size) for word in list_of_words]
    return np.mean(word_vectors, axis=0)



# getting gensim stopwords

stopwords = gensim.parsing.preprocessing.STOPWORDS
stopwords = stopwords.difference({'computer'})  # removing the word computer from stopwords


# preprocess the document to get list of tokens
def preprocess(document):
    string = re.sub('[^\w\d\s\$]','',document)
    tokens = (string.lower()).split(' ')  # Output of this step is list of words
    tokens = [word for word in tokens if word]
    tokens = [word for word in tokens if word not in stopwords and len(word)>= 2 ]
    return tokens


def tagging(corpus):
    for i, line in enumerate(corpus):
        yield TaggedDocument(line, [i])
        
        
def count_vectorizer(text,stopwords = stopwords):
    Vectorizer = CountVectorizer(
        max_features=100,
        binary=True,
    )
    vecs = Vectorizer.fit_transform(text)
    vecs = pd.DataFrame(vecs.toarray(), columns=Vectorizer.get_feature_names_out(), index=text.index)
    vecs = vecs.add_prefix('col_')
    return vecs


def calculate_metrics(y_true, y_pred, print_bool=False):
    cm = {}

    # Calculating classification metrics using sklearn
    cm['recall'] = metrics.recall_score(y_true, y_pred)
    cm['precision'] = metrics.precision_score(y_true, y_pred)
    cm['accuracy'] = metrics.accuracy_score(y_true, y_pred)

    # Confusion matrix components for TPR and TNR
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    cm['tpr'] = tp / (tp + fn)  # True Positive Rate
    cm['tnr'] = tn / (tn + fp)  # True Negative Rate

    if print_bool:
        print('Recall:', cm['recall'])
        print('Precision:', cm['precision'])
        print('Accuracy:', cm['accuracy'])
        print('TPR:', cm['tpr'])
        print('TNR:', cm['tnr'])

    return cm

def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print("Cross-validated scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())

    # Fit the model and make predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, print_bool=True)
    return metrics
