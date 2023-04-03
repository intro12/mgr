import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 10,7
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)

from sklearn.naive_bayes import MultinomialNB

data_train=pd.read_csv("poker-hand-training-true.data",header=None)
data_test=pd.read_csv("poker-hand-testing.data",header=None)

col=['Suit of card #1','Rank of card #1','Suit of card #2','Rank of card #2','Suit of card #3','Rank of card #3','Suit of card #4','Rank of card #4','Suit of card #5','Rank of card 5','Poker Hand']

data_train.columns=col
data_test.columns=col

y_train=data_train['Poker Hand']
y_test=data_test['Poker Hand']

x_train=data_train.drop('Poker Hand',axis=1)
x_test=data_test.drop('Poker Hand',axis=1)

print('Shape of Training Set:',x_train.shape)
print('Shape of Testing Set:',x_test.shape)

model = ComplementNB()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

acc=accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred, average="weighted")

print("Accuracy:", acc)
print("F1 Score:", f1)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
