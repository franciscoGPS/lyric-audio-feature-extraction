from keras.models import Sequential
from keras.layers import Activation
from data_utils import load_CIFAR10
model = Sequential()

#cifar10_dir = 'datasets/cifar-10-batches-py'
#X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
	optimizer='sgd',
	metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32)
