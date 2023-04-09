import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from matplotlib import pyplot as plt

# Define custom data augmentation function
def salt_pepper_noise(img):
    cv2.resize(img, (128, 128))
    s_vs_p = 0.5
    amount = 0.05
    out = np.copy(img)
    # Salt mode
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = [np.random.randint(0, i-1, int(num_salt))
              for i in img.shape]
    out[coords] = 255
    # Pepper mode
    num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i-1, int(num_pepper))
              for i in img.shape]
    out[coords] = 0
    return out


# prepare dataset
train_dir = '/Users/busragul/Desktop/DeepLearning_HW2_BusraGUL/content/training/data'
test_dir = '/Users/busragul/Desktop/DeepLearning_HW2_BusraGUL/content/test/data'


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_dir,
                                                    color_mode="grayscale",
                                                    target_size=(128, 128),
                                                    batch_size=32,
                                                    class_mode='categorical')


testing_set = test_datagen.flow_from_directory(test_dir,
                                                  color_mode= "grayscale",
                                                  target_size=(128, 128),
                                                  batch_size=32,
                                                  class_mode='categorical')

# Create AlexNet model with modified input layer

model = Sequential()
model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(128,128,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='softmax'))


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.001), metrics=['accuracy'])
model.summary()


history = model.fit(training_set,
         steps_per_epoch=len(training_set),
         epochs=50,
         validation_data=testing_set,
         validation_steps=len(testing_set))


plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('loss')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()