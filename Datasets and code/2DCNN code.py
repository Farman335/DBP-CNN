# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:25:26 2018

@author: Khanh Lee
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:08:49 2018

@author: Khanh Lee
"""

import numpy
# fix random seed for reproducibility
from sklearn.metrics import accuracy_score, \
    log_loss, \
    classification_report, \
    confusion_matrix, \
    roc_auc_score, \
    average_precision_score, \
    auc, \
    roc_curve, f1_score, recall_score, matthews_corrcoef, auc

seed = 123
numpy.random.seed(seed)
# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.layers import Activation
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import h5py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# define params
trn_file = 'PSSM_DCT_14189_CNN.csv'

nb_classes = 2
nb_kernels = 3
nb_pools = 2

# load training dataset
dataset = numpy.loadtxt(trn_file, delimiter = ",")  # , ndmin = 2)
# split into input (X) and output (Y) variables
X = dataset[:,1:401].reshape(len(dataset),1,20,20)
Y = dataset[:, 0]


def PrintMean(nsplit,Acc_):
    print('Accuracy: {0:.2f}%\n'.format((float(Acc_)/nsplit) * 100.0))
    

def cnn_model():
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape = (1, 20, 20)))
    model.add(Conv2D(4, nb_kernels, nb_kernels, activation = 'relu'))
    model.add(MaxPooling2D(strides = (nb_pools, nb_pools), dim_ordering = 'th'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(8, nb_kernels, nb_kernels, activation = 'relu'))
    model.add(MaxPooling2D(strides = (nb_pools, nb_pools), dim_ordering = 'th'))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(16, nb_kernels, nb_kernels, activation = 'relu'))
    model.add(MaxPooling2D(strides = (nb_pools, nb_pools), dim_ordering = 'th'))

    ## add the model on top of the convolutional base
    model.add(Flatten())
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('sigmoid'))

    # model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=['accuracy'])
   # model.compile(loss = 'binary_crossentropy', optimizer = "adadelta", metrics = ['accuracy'])
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    # Compile model
    return model


# define 5-fold cross validation test harness
#kfold = KFold(n_splits = 5, shuffle = True, random_state = seed)
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
#cvscores = []
model = cnn_model()
Acc = 0
Sen = 0
Sp = 0
Precision=0
F1 = 0
MCC = 0
AUC = 0
ExCM=np.array([
        [0, 0],
        [0, 0],
    ], dtype = int)
for train, test in kfold.split(X, Y):
    # model = cnn_model()
    # fit the model
    model.fit(X[train], np_utils.to_categorical(Y[train], nb_classes), epochs = 100, batch_size = 20, verbose = 1)
    # evaluate the model
    scores = model.evaluate(X[test], np_utils.to_categorical(Y[test], nb_classes), verbose = 0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    #cvscores.append(scores[1] * 100)
    # prediction
    true_labels = numpy.asarray(Y[test])
    predictions = model.predict_classes(X[test])

    auROC = []

    AUC = []
    mean_TPR = 0.0
    mean_FPR = np.linspace(0, 1, 100)

    CM = np.array([
        [0, 0],
        [0, 0],
    ], dtype = int)
    y_proba = model.predict_proba(X[test])[:, 1]

    FPR, TPR, _ = roc_curve(true_labels, y_proba)
    mean_TPR += np.interp(mean_FPR, FPR, TPR)
    mean_TPR[0] = 0.0
    roc_auc = auc(FPR, TPR)
    #y_artificial = model.predict(X[test])
    auROC.append(roc_auc_score(y_true = true_labels, y_score = y_proba))
    AUC.append(roc_auc)
    print('AUC: {0:.2f}%'.format(np.mean(AUC) * 100.0))
    CM = confusion_matrix(y_pred = predictions, y_true = true_labels)
    ExCM+=CM
    print(str(ExCM))

    TN, FP, FN, TP = CM.ravel()


    Acc += round((TP + TN) / (TP + FP + TN + FN),4)
    Sen += round((TP / (TP + FN)),4)
    Sp += round(TN / (FP + TN),4)
    Precision += round(TP / (TP + FP), 4)
    MCC += round(((TP * TN) - (FP * FN)) / (np.math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))),4)

    F1 += round((2 * TP) / ((2 * TP) + FP + FN),4)

PrintMean(5, Acc)



'''# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")'''

FPR, TPR, _ = roc_curve(true_labels, y_proba)
mean_auc = auc(mean_FPR, mean_TPR)
# mean_TPR[-1] = 1.0
plt.plot(
    FPR,  # mean_FPR,
    TPR,
    # mean_TPR,
    linestyle = '-',
    label = '{} ({:0.3f})'.format('AUC=', mean_auc), lw = 2.0)

plt.plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'k', label = 'Random')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize = 12)
plt.ylabel('True Positive Rate (TPR)', fontsize = 12)
plt.title('ROC curve', fontweight = 'bold')
plt.legend(loc = 'lower right')
plt.savefig('auROC.png', dpi = 300)
plt.show()

