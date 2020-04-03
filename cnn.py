# Convolutional Neural Network
#no preprocessing. It is done manually by making seperate folders.

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#3 in input shape for no. of channels in image = here rgb so 3.
#activation fn used so that no negative pixel value and for non-linearity.
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer to get better accuracy.
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
#power of 2 preferable and by experiment around 100 should be the value.
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#image augmentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, #feature scaling all between 0 and 1
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,#equal to no. of training set
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000)#equal to size of test set

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)#this will make image as 3d array. 
test_image = np.expand_dims(test_image, axis = 0)#we need 4 d array for conv2d it corresponds to the batch
result = classifier.predict(test_image)
training_set.class_indices#tells us whether 1 is dog
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
