from keras.models import Sequential
from keras.layers import Conv2D,Activation,BatchNormalization,MaxPooling2D,Flatten,Dense
import numpy as np

input_shape = (64,64,3)
num_classes = 2
#创建CNN模型
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3)),activation='relu')
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=())

x_train = np.random.rand(100,*input_shape)
y_train = np.random.randint(0,2,size=(100,num_classes))

model.fit(x_train,y_train,epochs=200)

def residual_block(input,filters):
    x=Conv2D(filters=filters,kernel_size=(3,3),padding='same')(input)
    x=BatchNormalization(x)
    x=Conv2D(filters=filters,kernel_size=(3,3),padding='same')(x)
    x=BatchNormalization(x)
    x=Activation('relu')(x)

    return x