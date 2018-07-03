import numpy as np
import os
import pickle
import cPickle

def load_data_shuffle(cv):

    train_pos_save = "data/VS/Data_not_token/Fold_"+ str(cv)+"/train_pos.npy"
    train_neg_save = "data/VS/Data_not_token/Fold_"+ str(cv)+"/train_neg.npy"
    train_neu_save = "data/VS/Data_not_token/Fold_"+ str(cv)+"/train_neu.npy"

    test_pos_save = "data/VS/Data_not_token/Fold_"+ str(cv)+"/test_pos.npy"
    test_neg_save = "data/VS/Data_not_token/Fold_"+ str(cv)+"/test_neg.npy"
    test_neu_save = "data/VS/Data_not_token/Fold_"+ str(cv)+"/test_neu.npy"




    #Load train data
    pos_train = np.load(train_pos_save)
    neg_train = np.load(train_neg_save)
    neu_train = np.load(train_neu_save)

    y_pos_train = []
    for i in pos_train:
        y_pos_train.append([1,0,0])
    y_pos_train = np.array(y_pos_train)

    y_neg_train = []
    for i in neg_train:
        y_neg_train.append([0, 1, 0])
    y_neg_train = np.array(y_neg_train)

    y_neu_train = []
    for i in neu_train:
        y_neu_train.append([0, 0, 1])
    y_neu_train = np.array(y_neu_train)


    #load test data
    pos_test = np.load(test_pos_save)
    neg_test = np.load(test_neg_save)
    neu_test = np.load(test_neu_save)

    y_pos_test = []
    for i in pos_test:
        y_pos_test.append([1,0,0])
    y_pos_test = np.array(y_pos_test)

    y_neg_test = []
    for i in neg_test:
        y_neg_test.append([0, 1, 0])
    y_neg_test = np.array(y_neg_test)

    y_neu_test = []
    for i in neu_test:
        y_neu_test.append([0, 0, 1])
    y_neu_test = np.array(y_neu_test)


    #split train and validate set
    val_len = len(pos_train)/10

    pos_val = pos_train[0:val_len]
    pos_train= pos_train[val_len:]
    y_pos_val = y_pos_train[0:val_len]
    y_pos_train = y_pos_train[val_len:]


    neg_val = neg_train[0:val_len]
    neg_train = neg_train[val_len:]
    y_neg_val = y_neg_train[0:val_len]
    y_neg_train = y_neg_train[val_len:]


    neu_val = neu_train[0:val_len]
    neu_train = neu_train[val_len:]
    y_neu_val = y_neu_train[0:val_len]
    y_neu_train = y_neu_train[val_len:]



 


    X_train = np.concatenate([pos_train,neu_train,neg_train])
    y_train = np.concatenate([y_pos_train,y_neu_train,y_neg_train])

    X_val = np.concatenate([pos_val,neu_val,neg_val])
    y_val = np.concatenate([y_pos_val,y_neu_val,y_neg_val])

    X_test = np.concatenate([pos_test,neu_test,neg_test])
    y_test = np.concatenate([y_pos_test,y_neu_test,y_neg_test])

    # print X_train.shape, y_train.shape
    # print X_test.shape, y_test.shape
    # print X_val.shape, y_val.shape

    return X_train, y_train, X_test, y_test, X_val, y_val




