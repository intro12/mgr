import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 10,7
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)
from catboost import CatBoostClassifier

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

model = CatBoostClassifier(task_type='GPU',classes_count=10,num_trees=10000,objective='MultiClass',max_depth=10,verbose=0)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test,y_pred, average="weighted")

print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("F1 Score:", f1)