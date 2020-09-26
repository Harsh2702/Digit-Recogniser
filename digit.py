
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'D:\\datasets\train.csv')
#data = data.head(1000)
#print(data.shape)

y = data['label']
x = data.drop(data.columns[0],axis = 1)

from sklearn.model_selection import train_test_split
xtrain ,xtest, ytrain, ytest =  train_test_split(x,y,test_size = 0.1)
xtrain = xtrain.values
ytrain = ytrain.values
xtest = xtest.values
ytest = ytest.values
xtrain = xtrain/256
xtest = xtest/256
#training
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(solver= 'adam',activation = 'relu', hidden_layer_sizes = (128,128))
model.fit(x,y)

rr = model.predict(xtest)

cnt = 0
for i in range(0,len(rr)):
    if rr[i] == ytest[i]:
        cnt = cnt+1
        
        
print("accuracy ",cnt/len(rr))

#testing from dataset
d = xtest[4]
d.shape = (28,28)
plt.imshow(255-d,cmap = 'gray')
plt.show()

t =model.predict([xtest[4]])
print(t)

#testing from external image
import cv2
img = cv2.imread('e1.jpg',0)
img1 = cv2.resize(img,(28,28))


import numpy
dd = numpy.array(img1)

dd.shape = (28,28)
plt.imshow(dd,cmap = 'gray')
plt.show()

#print(dd.shape)

s = dd.reshape(1,-1)
print('your given number is ')
print(model.predict(s))

