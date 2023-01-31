import numpy as np
import keras.utils as image
from keras.models import load_model
import sys
from keras.preprocessing.image import ImageDataGenerator
batch_size = 32
input_size = (300, 300)
model = load_model('model_final1.h5')


test_image = image.load_img('detect.jpg', target_size= input_size)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
                                    
training_set = train_datagen.flow_from_directory('dataset/train',
                                                    target_size = input_size,
                                                    batch_size = batch_size,
                                                    class_mode = 'categorical')

count = 0
for i in range(len(result[0])):
    if result[0][i]==0:
        count+=1
    if result[0][i]==1:
        break

x = training_set.class_indices
print(list(x.keys())[list(x.values()).index(count)])

