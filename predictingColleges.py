import re
import csv
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import callbacks
from keras.preprocessing import sequence
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from cnnlstmmodel import loaded_model as model

result = {}

colleges = ["iitk", "iitm", "iitd" ,"iitb","iitkgp","cbit" , "jntuh", "jntuk", "nitwarangal","iiith","uce", "vnrvjiet", "snist","mlrit" ]

for clg in colleges :
    test_phrase = []

    with open("D:/RankingColleges/"+clg+".csv") as testing:
        test = csv.reader(testing, delimiter="\t", quotechar='"')
        for s in test:
            test_phrase.append(s)

    def clean_phrase(phrase):
        #Remove punctuation (with a regular expression) and convert to lower case
        words = (re.sub("[^a-zA-Z]", " ", str(phrase))).lower()
        return words

    # run preprocessing function  on test dataset
    test_clean_phrases = []
    for xw in test_phrase:
        new_test = clean_phrase(xw)
        test_clean_phrases.append(new_test)

    test_all_text=' /n '.join(test_clean_phrases)


    # split each reviews of the training dataset and join them as a string
    test_reviews = test_all_text.split(' /n ')
    test_all_text = ' '.join(test_reviews)
    # split each word of the training dataset in the string to a list
    test_words = test_all_text.split()

    # combine the list that contains the individual words in the datasets
    full_words = test_words


    #create dictionaries that map the words in the vocabulary to integers. 
    #Then we can convert each of our reviews into integers so they can be passed into the network.
    from collections import Counter
    counts = Counter(full_words)
    vocab = sorted(counts, key=counts.get, reverse=True)

    #Build a dictionary that maps words to integers
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    test_reviews_ints = []
    for eachs in test_reviews:
        test_reviews_ints.append([vocab_to_int[word] for word in eachs.split( )])

    max_review_length = 32
    x_test = sequence.pad_sequences(test_reviews_ints, maxlen=max_review_length)

    print(x_test.shape)

    # run prediction
    test_pred = model.predict_classes(x_test)

    print("+++++++++++++++++++++++++++++")
    #print(test_pred[1234])

    print(len(x_test))
    print(len(test_pred))
    # edits the test file to input the prediction labels

    neg1 = test_pred.tolist().count(0)
    neu1 = test_pred.tolist().count(1)
    pos1 = test_pred.tolist().count(2)

    tot = neg1+neu1+pos1
    val = [None]*3
    val[0] = neg1
    val[1] = neu1
    val[2] = pos1 

    result[clg] = val  




    '''
    d = {"Tweets" : x_test , "Sentiment" : test_pred}


    test_df = pd.DataFrame()
    test_df["Tweets"] = x_test.tolist()
    test_df["Sentiment"] = test_pred.tolist()

    print(test_df)

    '''





