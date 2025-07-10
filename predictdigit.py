import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
#Load MNIST dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data()
#2.normalize pixel values to [0,1]
x_train=x_train/255.0
x_test=x_test/255.0
#3.label the outcome if digit is 4 => 1 or 0
y_train_binary=np.isin(y_train,[4])
y_test_binary=np.isin(y_test,[4])
#4.create a model
model=Sequential([Flatten(input_shape=(28,28)),Dense(64,activation='relu'),Dense(1,activation='sigmoid')])
#5.compile model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#6.train the model
model.fit(x_train,y_train_binary,epochs=10,validation_data=(x_test,y_test_binary))
#7.predict the model with sample
index=6
sample=x_test[index]
pred=model.predict(sample.reshape(1,28,28),verbose=0)
pred_val=int(pred[0][0]>=0.5)
print(f"Actual digit:{y_test[index]}")
print(f"predicted:{pred[0][0]}")
#8.show the image
plt.imshow(sample,cmap='gray')
plt.title("predict 4")
plt.show()