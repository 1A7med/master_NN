from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.callbacks import TensorBoard

import keras as kr
from keras.models import load_model

name = '3c_dm'

log = 'tb/' + name
mdl =  'models/' + name + '.h5'

bs = 100
num_classes = 10
epochs = 20
tb = TensorBoard(log_dir=log, histogram_freq=0, batch_size=bs, write_graph=True, write_grads=False, embeddings_freq=0, update_freq='epoch')

# input image dimensions
img_x, img_y = 28, 28

# load the MNIST data set, which already splits into train and test sets for us
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
# because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = kr.utils.to_categorical(y_train, num_classes)
y_test = kr.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add( Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

opt = kr.optimizers.Adadelta()

model.compile( loss=kr.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'] )
model.fit(x_train, y_train, batch_size=bs, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[tb] )

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save(mdl)
