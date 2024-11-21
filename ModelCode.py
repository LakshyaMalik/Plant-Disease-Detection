from google.colab import drive 
drive.mount('/content/drive') 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 

##tranining image preprocessing 
training_set = tf.keras.utils.image_dataset_from_directory( 
 directory="/content/drive/MyDrive/project data/plantds/train", 
 labels="inferred", 
 label_mode="categorical", 
 class_names=None, 
 color_mode="rgb", 
 batch_size=32, 
 image_size=(256,256), 
 shuffle=True, 
 seed=None, 
 validation_split=None, 
 subset=None, 
 interpolation="bilinear", 
 follow_links=False, 
 crop_to_aspect_ratio=False, 
) 

##validation image preprocesing 
validation_set = tf.keras.utils.image_dataset_from_directory( 
 directory="/content/drive/MyDrive/project data/plantds/valid", 
 labels="inferred", 
 label_mode="categorical", 
 class_names=None, 
 color_mode="rgb", 
 batch_size=32, 
 image_size=(256,256), 
 shuffle=True, 
 seed=None, 
 validation_split=None, 
 subset=None, 
 interpolation="bilinear", 
 follow_links=False, 
 crop_to_aspect_ratio=False 
) 

from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout 
from tensorflow.keras.models import Sequential 
model = Sequential() 

##BUILDING CONVulation LAYER1 
model.add(Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[256,256,3])) 
model.add(Conv2D(filters=32,kernel_size=3,activation='relu')) 
model.add(Conv2D(filters=32,kernel_size=3,activation='relu')) 
model.add(MaxPooling2D(pool_size=2,strides=2)) 

##BUILDING CONVulation LAYER2 
model.add(Conv2D(filters=64,kernel_size=3,activation='relu',input_shape256,256,3])) 
model.add(Conv2D(filters=64,kernel_size=3,activation='relu')) 
model.add(Conv2D(filters=64,kernel_size=3,activation='relu')) 
model.add(MaxPooling2D(pool_size=2,strides=2)) 

##BUILDING CONVulation LAYER3 
model.add(Conv2D(filters=128,kernel_size=3,activation='relu',input_shape=[256,256,3])) 
model.add(Conv2D(filters=128,kernel_size=3,activation='relu')) 
model.add(Conv2D(filters=128,kernel_size=3,activation='relu')) 
model.add(MaxPooling2D(pool_size=2,strides=2)) 

##BUILDING CONVulation LAYER4 
model.add(Conv2D(filters=256,kernel_size=3,activation='relu',input_shape=[256,256,3])) 
model.add(Conv2D(filters=256,kernel_size=3,activation='relu')) 
model.add(Conv2D(filters=256,kernel_size=3,activation='relu')) 
model.add(MaxPooling2D(pool_size=2,strides=2)) 

##Add dropout 
model.add(tf.keras.layers.Dropout(0.2)) 

##Flatten layer 
model.add(Flatten()) 
model.add(Dense(units=1024,activation='relu')) 

#Output Layer 
model.add(Dense(units=4,activation='softmax')) 

##Compiling Model 
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='categoric
al_crossentropy',metrics=['accuracy']) 
model.summary() 

training_history = model.fit(x=training_set, validation_data=validation_set, epochs=52) 

##Model evaluation on training set 

training_loss, training_acc = model.evaluate(training_set) 
val_loss, val_acc = model.evaluate(validation_set) 

model.save("iteration2.keras") 

#record_history 
import json 
with open('training_history.json', 'w') as f: 
 json.dump(training_history.history,f) 

epochs = list(range(1, 53)) 
plt.plot(epochs, training_history.history['accuracy'],color='green',label='Training 
Accuracy') 
plt.plot(epochs, training_history.history['val_accuracy'],color='blue',label='Validation 
Accuracy') 
plt.title('Visualization of Accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend() 
plt.show() 

##test image preprocesing 
test_set = tf.keras.utils.image_dataset_from_directory( 
 directory="/content/drive/MyDrive/project data/plant_ds/valid/", 
 labels="inferred", 
 label_mode="categorical", 
 class_names=None, 
 color_mode="rgb", 
 batch_size=32, 
 image_size=(256,256), 
 shuffle=False, 
 seed=None, 
 validation_split=None, 
 subset=None, 
 interpolation="bilinear", 
 follow_links=False, 
 crop_to_aspect_ratio=False 
) 

import numpy as np 
from sklearn.metrics import confusion_matrix 

# Generate predictions 
y_pred = model.predict(test_set) 
y_pred_classes = np.argmax(y_pred, axis=1) 
y_true = np.concatenate([np.argmax(y, axis=1) for x, y in test_set], axis=0) 

# Confusion Matrix 
cm = confusion_matrix(y_true, y_pred_classes) 
plt.figure(figsize=(8, 6)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') 
plt.xlabel('Predicted') 
plt.ylabel('Actual') 
plt.title('Confusion Matrix Heatmap') 
plt.show() 

# Precision Report 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, 
classification_report 
import numpy as np 
y_pred = model.predict(test_set) 
y_pred_classes = np.argmax(y_pred, axis=1) 
y_true = np.concatenate([np.argmax(y, axis=1) for x, y in test_set], axis=0) 
 
# Classification Report 
accuracy = accuracy_score(y_true, y_pred_classes) 
print(f"Accuracy: {accuracy}") 
print("Classification Report:") 
print(classification_report(y_true, y_pred_classes)) 
