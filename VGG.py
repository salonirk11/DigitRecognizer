
import numpy as np
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import applications
from keras.preprocessing.image ImageDataGenerator

img_rows, img_cols, img_channel = 224, 224, 3


#preprocess


image_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))
add_model = Sequential()
add_model.add(Flatten(input_shape=image_model.output_shape[1:]))
add_model.add(Dense(256, activation='relu'))
add_model.add(Dense(1, activation='sigmoid'))

model = Model(inputs=image_model.input, outputs=add_model(image_model.output))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.summary()

train_data = ImageDataGenerator(
            rotation=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

model.fit_generator(train_data.flow(x_train,y_train,batch_size=batch_size),
                      batch_size=batch_size,epochs=50,validation_data=(x_test,y_test))

pred=model.predict(test)

