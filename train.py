import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras import optimizers

import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

train= ImageDataGenerator(validation_split=0.2,rescale=1./255)
train_dataset=train.flow_from_directory('dataset/train/',subset='training',target_size=(300,300),batch_size=30,class_mode="categorical")
validation_dataset=train.flow_from_directory('dataset/train/',subset='validation',target_size=(300,300),batch_size=30,class_mode="categorical")
print(train_dataset)
#train_dataset.classes= to_categorical(train_dataset.classes).astype(int)
#validation_dataset.classes= to_categorical(validation_dataset.classes).astype(int)
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(300, 300, 3)),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                                    tf.keras.layers.MaxPool2D(2, 2),
                                    
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation="relu"),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(26, activation="softmax")

                                    ])
model.compile(optimizer ="adam" , loss="categorical_crossentropy", metrics=['accuracy'])
model_fit=model.fit(train_dataset,epochs=10,validation_data=validation_dataset)

# fit the cnn model to the trainig set and testing it on the test set

model.save("model_final1.h5")
