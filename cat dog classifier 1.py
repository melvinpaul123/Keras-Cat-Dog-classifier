import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

X = []
Y=[]
labels = ["cats","dogs"]
cur_path = os.getcwd()
imgs= cur_path+"/dogscats/images"
print("Loading Images and Labels...")
#Retrieving the images and their labels 
for label in labels:
    i=labels.index(label)
    path = os.path.join(imgs,label)
    images = os.listdir(path)
    for a in images:
        try:
            image = cv2.imread(imgs +"/"+label+"/"+a)
            image = cv2.resize(image,(100,100))
            image = np.array(image)
            image=image/255
            X.append(image)
            Y.append(i)
        except:
            print("Error loading image")

X=np.array(X)
Y=np.array(Y)
print("Loading completed")
print("No. of Images = %d, No. of Labels= %d" %(X.shape[0], Y.shape[0]))
#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
print("Spliting images for training and testing")
print("No. of Training Images = %d, No. of Training Labels= %d" %(X_train.shape[0], y_train.shape[0]))
print("No. of Testing Images = %d, No. of Testing Labels= %d" %(X_test.shape[0], y_test.shape[0]))

del X
del Y

#Converting the labels into one hot encoding
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

#Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3,4), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(2, activation='softmax'))

#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 10
reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto')
mc = ModelCheckpoint('modelcatanddog_autokeras.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=10)
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test),callbacks=[early_stop, reduce, mc])
best_model= load_model('modelcatanddog_autokeras.h5')
best_model.save("catdogclassifier.h5")
best_model.summary()

#plotting graphs for accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.title('Acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#testing accuracy on test dataset
from sklearn.metrics import accuracy_score

X_val =[]
Y_val=[]
labels = ["cats","dogs"]
imgs1= cur_path+"/dogscats/valid"

#Retrieving the images and their labels 
for label in labels:
    i=labels.index(label)
    path = os.path.join(imgs1,label)
    images = os.listdir(path)
    for a in images:
        try:
            image = cv2.imread(imgs1 +"/"+label+"/"+a)
            image = cv2.resize(image,(100,100))
            image = np.array(image)
            image=image/255
            X_val.append(image)
            Y_val.append(i)
        except:
            print("Error loading image")

X_val=np.array(X_val)
Y_val=np.array(Y_val)

pred = best_model.predict_classes(X_val)

#Accuracy with the test data
print("Accuracy Score = %f" %((accuracy_score(Y_val, pred))*100)+"%")