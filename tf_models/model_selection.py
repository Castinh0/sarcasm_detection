import pandas as pd
import numpy as np
import time

from stop_words import get_stop_words

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, f1_score, accuracy_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, LSTM, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

"""
-------HYPERPARAMETERS--------
remove_stop_words = False
remove_punctuations = True
to_lower = True
vocab_size = 1000
embedding_dim = 64
sentence_length_threshold = -1
learning_rate = 0.0005

------CONSTANT------
trunc_type='post'
padding_type='post'
optimizer = Adam
loss_function = binary_cross_entropy
train_metric = Accuracy
epoch = 30
early_stopping_patience = 3
Model Architecture

-----METRICS-----
Train Accuracy
Train Loss
Test Accuracy
Test Loss
Test F1
Test AUC
Train Time
"""


def train(model_type, remove_stop_words, remove_punctuations, to_lower, vocab_size, embedding_dim, sentence_length_threshold, learning_rate, verbose = False):
    
    trunc_type='post'
    padding_type='post'
    optimizer = Adam(learning_rate=learning_rate)
    loss_function = 'binary_crossentropy'
    train_metric = "accuracy"
    num_epochs = 30
    early_stopping_patience = 3

    df = pd.read_csv("../data/corpora_dataset.csv",sep=";", encoding='utf8')

    if remove_stop_words:

        titles = df['titles'].to_list()

        stop = get_stop_words('italian')

        for i in range(len(titles)):

            titles[i]  = ' '.join([i for i in titles[i].split() if i not in stop])

        df["titles"] = titles

    sentences = df['titles'].values
    labels = df['Label'].values

    training_sentences, testing_sentences, training_labels, testing_labels = train_test_split(sentences, labels, test_size=0.2)

    #print(f"Traning Set Size: {len(training_sentences)}")
    #print(f"Test Set Size: {len(testing_sentences)}")

    if remove_punctuations:
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>", lower=to_lower)
    else:
        tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>", lower=to_lower, filters='')

    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    #print(f"\nTraining Set Word Count: {len(word_index)}")

    lengths = []

    for sentences in training_sentences:

        sen_len = len(sentences.split())

        lengths.append(sen_len)

    #print(f"\nSentences;\nMin Length: {min(lengths)}\nMax Length: {max(lengths)}\nMean Length: {(sum(lengths) / len(lengths))}\nSTD of Lengths: {np.std(lengths)}")
    
    if sentence_length_threshold != -1:
        #print(f"\nSentence Lenght Threshold overrided: {sentence_length_threshold}")
        pass
    else:
        sentence_length_threshold = int((sum(lengths) / len(lengths)) + (3 * np.std(lengths)))
        #print(f"\nEven if data has not normal distribution but we can take sentence length_threshold as Mean+3std: {sentence_length_threshold}")

    count = 0

    for i in lengths:

        if i <= sentence_length_threshold:
            count+=1

    #print(f"\nWe cover {(count/len(lengths)*100):.2f}% of the sentences without losing an information. Expectation of normal distribution is 99.7%.")

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=sentence_length_threshold, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=sentence_length_threshold, padding=padding_type, truncating=trunc_type)

    training_padded = np.array(training_padded)
    training_labels = np.array(training_labels)
    testing_padded = np.array(testing_padded)
    testing_labels = np.array(testing_labels)

    if model_type == "cnn":
        
        model = Sequential([
            Embedding(vocab_size, embedding_dim),
            Conv1D(16, 1, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(8, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
    if model_type == "lstm":
        
        model = Sequential([
            Embedding(vocab_size, embedding_dim),
            LSTM(16, return_sequences=False),
            Dense(8, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        
    if model_type == "dense":
        
        model = Sequential([
            Embedding(vocab_size, embedding_dim),
            Flatten(),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
     

    model.compile(loss=loss_function, optimizer=optimizer, metrics=[train_metric])

    early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True)
    
    #print()
    
    #print("Train Started.")

    start_time = time.time()

    history = model.fit(
        training_padded, training_labels,
        epochs=num_epochs,
        validation_data=(testing_padded, testing_labels),
        callbacks=[early_stopping],
        verbose=0
    )

    end_time = time.time()
    
    #print("Train Ended.")
    
    training_time = (end_time - start_time)

    y_pred_prob = model.predict(testing_padded)

    auc_score = roc_auc_score(testing_labels, y_pred_prob)

    fpr, tpr, thresholds = roc_curve(testing_labels, y_pred_prob)

    best_index = np.argmax(tpr-fpr)
    best_threshold = thresholds[best_index]
    best_tpr = tpr[best_index]
    best_fpr = fpr[best_index]

    #print(f"\nWe used cut off threshold: {best_threshold:.2f}")

    y_pred = (y_pred_prob > best_threshold).astype(int)

    train_accuracy = history.history['accuracy'][-1]
    test_accuracy = history.history['val_accuracy'][-1]
    train_loss = history.history['loss'][-1]
    test_loss = history.history['val_loss'][-1]
    f1 = f1_score(testing_labels, y_pred)
    
    if verbose:
        print("-----------METRICS-----------------")
        print("\nTrain Accuracy Score:", train_accuracy)
        print("Test Accuracy Score:", test_accuracy)
        print("Train Loss Score:", train_loss)
        print("Test Loss Score:", test_loss)
        print("Test F1 Score:", f1)
        print("Test AUC Score:", auc_score)
        print("Train Time:", training_time)
        print("-----------------------------------")

    parameters = [remove_stop_words, remove_punctuations, to_lower, vocab_size, embedding_dim, sentence_length_threshold, learning_rate]
    results = [train_accuracy, test_accuracy, train_loss, test_loss, f1, auc_score, training_time]
    
    return parameters, results