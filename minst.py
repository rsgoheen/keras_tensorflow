from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import SGD

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels_oh = np_utils.to_categorical(train_labels)
test_labels_oh = np_utils.to_categorical(test_labels)

network = Sequential()
network.add(Dense(input_dim=train_images.shape[1],
                    output_dim=50,
                    init='uniform',
                    activation='tanh'))

network.add(Dense(50))                    
network.add(Dense(input_dim=50, 
                    output_dim=train_labels_oh.shape[1],
                    init='uniform',
                    activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
network.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

network.fit(train_images, 
           train_labels_oh, 
           nb_epoch=50, 
           batch_size=300, 
           verbose=1, 
           validation_split=0.1, 
           show_accuracy=True)

print(network.metrics_names)
print(network.evaluate(test_images, test_labels_oh, batch_size=16))
