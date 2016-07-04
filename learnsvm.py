import cv2
import numpy as np

a = [1, 0, 0]
b = [0, 1, 0]
ta = [2,0,0]
tb = [0,2,0]
tc = [0,0,2]
test = np.float32([ta,tb,tc])
train = np.float32([a, b])
# same length as the training data. 
# this labels the training data
label = np.float32([1,-1]) 

svm = cv2.SVM()
svm.train(train,label)

# predict_all for multiple test samples
# use predict for a single test sample
resp = svm.predict_all(test)
print resp





