import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = 10,7

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils.np_utils import to_categorical
import numpy as np
# fixed random seed for reproducibility
np.random.seed(0)

num_layers_in=512
num_layers_h1=256
num_layers_h2=256
num_layers_out=10
num_epochs=50
num_dropout=0.2
num_batch_size=128

data_train=pd.read_csv("poker-hand-training-true.data",header=None)
data_test=pd.read_csv("poker-hand-testing.data",header=None)

col=['Suit of card #1','Rank of card #1','Suit of card #2','Rank of card #2','Suit of card #3','Rank of card #3','Suit of card #4','Rank of card #4','Suit of card #5','Rank of card 5','Poker Hand']

data_train.columns=col
data_test.columns=col

y_train=data_train['Poker Hand']
y_test=data_test['Poker Hand']

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

x_train=data_train.drop('Poker Hand',axis=1)
x_test=data_test.drop('Poker Hand',axis=1)

print('Shape of Training Set:',x_train.shape)
print('Shape of Testing Set:',x_test.shape)

model=Sequential()
model.add(Dense(num_layers_in, activation='relu',input_dim=10))
#model.add(Dropout(rate=num_dropout))
model.add(Dense(num_layers_h1, activation='relu'))
#model.add(Dropout(rate=num_dropout))
model.add(Dense(num_layers_h2, activation='relu'))
#model.add(Dropout(rate=num_dropout))
model.add(Dense(num_layers_out, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history=model.fit(x_train,y_train, epochs=num_epochs, batch_size=num_batch_size,verbose=0,validation_data=(x_test,y_test), shuffle=True)

score=model.evaluate(x_test,y_test,batch_size=num_batch_size)

print(model.metrics_names)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()