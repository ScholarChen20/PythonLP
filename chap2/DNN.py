import numpy as np
from keras.models import Sequential
from keras.layers import Dense

input_dim=10
num_classes=3

model=Sequential()

model.add(Dense(64,activation='relu',input_shape=(input_dim,)))
model.add(Dense(32,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=())

x_train = np.random.rand(100,input_dim)
y_train = np.random.randint(0,2,size=(100,num_classes))

model.fit(x_train,y_train,epochs=10)
