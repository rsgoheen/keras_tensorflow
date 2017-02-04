from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD
import os

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels_oh = np_utils.to_categorical(train_labels)
test_labels_oh = np_utils.to_categorical(test_labels)

model = Sequential()
model.add(Dense(input_dim=train_images.shape[1],
                    output_dim=50,
                    init='uniform',
                    activation='tanh'))

model.add(Dense(50))                    
model.add(Dense(input_dim=50, 
                    output_dim=train_labels_oh.shape[1],
                    init='uniform',
                    activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_images, 
           train_labels_oh, 
           nb_epoch=50, 
           batch_size=300, 
           verbose=1, 
           validation_split=0.1, 
           show_accuracy=True)

print(model.metrics_names)
print(model.evaluate(test_images, test_labels_oh, batch_size=16))


model_json = model.to_json()
with open("mnist_model.json", "w") as json_file:
    json_file.write(model_json)

h5file = "mnist_model.h5"
model.save(h5file)    

print("Reloading model and running on test data.")

model_reload = load_model(h5file)
print(model_reload.evaluate(test_images, test_labels_oh, batch_size=16))



