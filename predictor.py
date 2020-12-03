from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np

# load model
model = load_model('catdogclassifier.h5')
# summarize model.
model.summary()
# load dataset
cur_path = os.getcwd()
imgs= cur_path+"/dogscats/"

x=[]
y=[]
images = os.listdir(cur_path+"/dogscats/samples")
for img in images:
    image = cv2.imread(cur_path+"/dogscats/samples/"+img)
    image = cv2.resize(image,(100,100))
    x.append(image)
    y.append(img)
    
x=np.array(x)
x=np.array(x, dtype="float") / 255.0
y=np.array(y)   
pred = model.predict_classes(x)
prob = model.predict_proba(x)
for i in range(len(x)):
    if pred[i] == 0:
        print("X=%s, Predicted=%s, Probability=%f" % (y[i], "CAT", prob[i][0]*100))
    elif pred[i] == 1:
        print("X=%s, Predicted=%s, Probability=%f" % (y[i], "DOG", prob[i][1]*100))