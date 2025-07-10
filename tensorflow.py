import tensorflow as tf
import numpy as np
x=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
y=np.array([0,0,0,1],dtype=np.float32)
#create model
model=tf.keras.Sequential([tf.keras.layers.Dense(units=1,input_shape=(2,),activation='sigmoid')])
#compile model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x,y,epochs=100,verbose=0)
#evaluate
loss,accuracy=model.evaluate(x,y,verbose=0)
print("Accuracy:",accuracy)
#predict
predicitons=model.predict(x)
for i in range(len(x)):
  print(f"x[i]->{predicitons[i][0]}->{int(predicitons[i][0]>0.7)}")