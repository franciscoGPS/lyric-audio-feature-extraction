#!/usr/bin/python

import sys, getopt
from keras.models import Sequential
from keras.layers import Activation


def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print( 'test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print( 'test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg.strip()
      elif opt in ("-o", "--ofile"):
         outputfile = arg.strip()

   #print( 'Input file is "', inputfile+'"')
   #print( 'Output file is "', outputfile+'"')

if __name__ == "__main__":
   main(sys.argv[1:])

def build_model(X_train, y_train):
	
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

def 