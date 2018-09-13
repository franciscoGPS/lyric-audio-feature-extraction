   #input_sequences = Input(shape=(1,hidden))
   #processed_sequences = TimeDistributed(audio_model)(input_sequences)
   #lstm_out = LSTM(32)(processed_sequences)
   #mergedOut = Concatenate()([audio_model.output,lyrics_model.output])





  
   """
   mfcc2_X_train, mfcc2_X_test, mfcc2_y_train, mfcc2_y_test = split_data(labels2, x_mfcc2, test_size )
   

   gtg_model_1 = Sequential()
   results['gtg'] = train_double_LSTM(gtg_X_train, gtg_y_train, gtg_model_1, hidden_size)
   results['gtg_test'] = test_model(gtg_X_test, gtg_y_test, gtg_model_1)
   
   mfcc_model_1a = Sequential()
   results['mfcc1'] = train_double_LSTM(mfcc1_X_train, mfcc1_y_train, mfcc_model_1a, hidden_size)
   results['mfcc1_test'] = test_model(mfcc1_X_test, mfcc1_y_test, mfcc_model_1a)

   
   mfcc_model2a = Sequential()
   results['mfcc2']  = train_double_LSTM(mfcc2_X_train, mfcc2_y_train, mfcc_model2a, hidden_size)
   results['mfcc2_test'] = test_model(mfcc2_X_test, mfcc2_y_test, mfcc_model2a)
   
   



   
   #alldata = np.vstack((mfcc1_X_train, mfcc1_y_train))
   mfcc1_X_trainT = mfcc1_X_train.reshape(mfcc1_X_train.shape[0],mfcc1_X_train.shape[2]).T
   mfcc2_X_trainT = mfcc2_X_train.reshape(mfcc2_X_train.shape[0],mfcc2_X_train.shape[2]).T
   mfcc2_X_testT = mfcc2_X_test.reshape(mfcc2_X_test.shape[0], mfcc2_X_test.shape[2]).T


   
 

   gtg_model_2b = Sequential()
   results['gtg_b'] = train_LSTM_model(gtg_X_train, gtg_y_train, gtg_model_2b, hidden_size)
   results['gtg_test_b'] = test_model(gtg_X_test, gtg_y_test, gtg_model_2b)
   
   mfcc_model_2b = Sequential()
   results['mfcc1_b'] = train_LSTM_model(mfcc1_X_train, mfcc1_y_train, mfcc_model_2b, hidden_size)
   results['mfcc1_test_b'] = test_model(mfcc1_X_test, mfcc1_y_test, mfcc_model_2b)

   
   mfcc_model_2b = Sequential()
   history = train_LSTM_model(mfcc2_X_train, mfcc2_y_train, mfcc_model_2b, hidden_size)
   results['mfcc2_test_b'] = test_model(mfcc2_X_test, mfcc2_y_test, mfcc_model_2b)
   
   plt.plot(history.history['loss'])
   plt.title('Bidirectional LSTM loss')
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend(['train', 'test'], loc='upper left')
   model.summary()
   #plt.show()
   plt.show()
   """

   
   #scikitmodel = create_LSTM_model(mfcc1_X_train, mfcc2_y_train, hidden_size) 
   #, fit_params = dict(callbacks=[tbCallBack, early_stopping])
   






   """
   # evaluate using 10-fold cross validation
   kfold = StratifiedKFold(n_splits=2, shuffle=True)
   results = cross_val_score(best_model, mfcc1_X_train, mfcc1_y_train, cv=kfold)
   print(results.mean())
   
  

   #early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

   pipe = pipeline.Pipeline([
       ('nn', KerasClassifier(build_fn=create_BLSTM_model, nb_epoch=100, batch_size=15,
                              validation_split=0.2, callbacks=[ tbCallBack]))
   ])


   pipe.fit(mfcc1_X_train , mfcc1_y_train )

   #pipe.steps.append(('nn', model))

   pred = pipe.predict_proba(mfcc1_X_test)[:, 0]



   directory = os.path.dirname(os.path.realpath(__file__))
   model_step = pipe.steps.pop(-1)[1]
   joblib.dump(pipe, os.path.join(directory, 'pipeline.pkl'))
   models.save_model(model_step.model, os.path.join(directory, 'model.h5'))
   """




   #***************************************
      #train_dcca(x_mfcc1.squeeze(), labels_gtg.squeeze(), bow_infered.squeeze(), labels_gtg.squeeze(), 2000, batch_size=40)


   
   

   ##evaluate_dataset( gtg_X_train, gtg_y_train, gtg_X_test , gtg_y_test)



   #history = train_LSTM_model(concat_gtg, labels_gtg, 50)
   """
   epochs = 50
   hidden_size = gtg_y_train.shape[1]

   history2, model = train_RNNLSTM_model(concat_mfcc_mixed_train, gtg_y_train, hidden_size, epochs)
   test_model(concat_mfcc_mixed_test, gtg_y_test, model)
   plot_results(history2, epochs)


   ipdb.set_trace()

   input_sequences = Input(shape=(1,hidden))
   processed_sequences = TimeDistributed(audio_model)(input_sequences)
   lstm_out = LSTM(32)(processed_sequences)
   mergedOut = Concatenate()([audio_model.output,lyrics_model.output])
   """

   
   #scikitmodel = create_LSTM_model(mfcc1_X_train, mfcc2_y_train, hidden_size) 
   #, fit_params = dict(callbacks=[tbCallBack, early_stopping])
   


   """

   # evaluate using 10-fold cross validation
   kfold = StratifiedKFold(n_splits=2, shuffle=True)
   results = cross_val_score(best_model, mfcc1_X_train, mfcc1_y_train, cv=kfold)
   print(results.mean())
   
  
   early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')

   pipe = pipeline.Pipeline([
       ('nn', KerasClassifier(build_fn=create_BLSTM_model, nb_epoch=100, batch_size=15,
                              validation_split=0.2, callbacks=[ tbCallBack]))
   ])

   pipe.fit(mfcc1_X_train , mfcc1_y_train )

   #pipe.steps.append(('nn', model))

   pred = pipe.predict_proba(mfcc1_X_test)[:, 0]



   directory = os.path.dirname(os.path.realpath(__file__))
   model_step = pipe.steps.pop(-1)[1]
   joblib.dump(pipe, os.path.join(directory, 'pipeline.pkl'))
   models.save_model(model_step.model, os.path.join(directory, 'model.h5'))
   
   """
concat_gtg = np.concatenate((x_gtg.squeeze(), bow_infered), axis=1)
   concat_gtg = concat_gtg.reshape(concat_gtg.shape[0], 1, concat_gtg.shape[1])
   concat_mfcc = np.concatenate((x_mfcc1.squeeze(), bow_infered), axis=1)
   if calculate_mixed:
      concat_mfcc = concat_mfcc.reshape(concat_mfcc.shape[0], 1, concat_mfcc.shape[1])
      concat_mfcc_mixed = np.concatenate((mfcc_mixed.squeeze(), bow_infered), axis=1)
      concat_mfcc_mixed = concat_mfcc_mixed.reshape(concat_mfcc_mixed.shape[0], 1, concat_mfcc_mixed.shape[1])

         """         
               if devide_mfcc:
                  for seq in range(len(mfcc_1)):
                     x_mfcc_1.append(np.array(mfcc_1[seq].reshape(1, mfcc_1[seq].size)).astype(np.float32))
                     labels_1.append(int(MOODS.index(mood)))
               else:"""
                  #x_gtg.append(np.array(gtg.reshape(1, gtg.size)).astype(np.float32))
                  #x_gtg.append(gtg.reshape(gtg.size,1))


   #gtg_X_train, gtg_X_test, gtg_y_train, gtg_y_test = split_data(labels_gtg, x_gtg, test_size )
      

   
      gtg_X_train, gtg_X_test, gtg_y_train, gtg_y_test, bow_X_train, bow_X_test, lyr_X_train, lyr_X_test, concat_mfcc_mixed_train, concat_mfcc_mixed_test = train_test_split(x_gtg, labels_gtg, bow_infered, lyrics_infered, concat_mfcc_mixed,  test_size=test_size )
   else:
         gtg_X_train, gtg_X_test, gtg_y_train, gtg_y_test, bow_X_train, bow_X_test, lyr_X_train, lyr_X_test = train_test_split(x_gtg, labels_gtg, bow_infered, lyrics_infered,  test_size=test_size )

   
   #mfcc_mixed = mfcc_mixed.reshape(mfcc_mixed.shape[0], 1, mfcc_mixed.shape[1])


   #train_deepcca(x_gtg, labels_gtg.squeeze(), bow_infered.squeeze(), labels_gtg.squeeze(), 30, batch_size=20)


   #evaluate_dataset(x_gtg, labels1, bow_infered.squeeze(), labels1, build_BLSTM_model)




   def test_model(X1_test, X2_test, y_test, model):
   
   correct = 0

   yhat = model.predict([X1_test, X2_test], y_test, verbose = 1)
   #labels_np = np.array(yhat).astype(dtype=np.uint8)
   #labels_onehot = (np.arange(19) == labels_np[:, None]).astype(np.float32)
   scores = model.evaluate([X1_test, X2_test], y_test, verbose = 1)
   for i in range(X_test.shape[0]):
      if(np.argmax(y_test[i]) == yhat[i]):
         print('Expected:', MOODS[y_test[i]], 'Predicted', MOODS[yhat[i]])
         correct+=1
         print()


   model.summary()
   accuracy = correct/X_test.shape[0]
   print("accuracy: %.6f" % accuracy*100 )
   print("Accuracy: %.2f%%" % (scores[1]*100))
   return accuracy
   
   


def train_double_LSTM(X, y, model, hidden_size):
   loss = list()
   print(X.shape)
   print(y.shape)
   
   model.add(LSTM(hidden_size, return_sequences=True, stateful=False, input_shape=(1, X.shape[2])))
   model.add(Dropout(0.5))
   #model.add(LSTM(hidden_size, return_sequences=False))
   #model.add(Dropout(0.2))
   model.add(Dense(19, activation='softmax'))
   model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])  
   history = model.fit(X, y, epochs=100, verbose=0, batch_size=20, callbacks=[tbCallBack], shuffle=True)

   # summarize history for loss
   plt.plot(history.history['loss'])
   """
   plt.title('double LSTM model loss')
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend(['train', 'test'], loc='upper left')
   """
   model.summary()
   #plt.show()
   return history

def audio_dnn(X, y):
   
   print(X.shape)
   print(y.shape)
   model = Sequential()
   model.add(Dense(1024, input_shape=(X.squeeze().shape[1],), activation="sigmoid" ))
   # now the model will take as input arrays of shape (*, 16)
   # and output arrays of shape (*, 32)

   #model.add(Activation('sigmoid'))
   model.add(Dense(1024, activation="sigmoid"))
   #model.add(Activation('sigmoid'))
   model.add(Dense(y.shape[1], activation='linear'))

   model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])  
   model.fit(X.squeeze(), y, epochs=10, verbose=0, batch_size=20, callbacks=[tbCallBack], shuffle=True)



   return model


def train_LSTM_model(X, y, hidden_size, epochs): 
   loss = list()
   model = Sequential()
   print(X.shape)
   print(y.shape)
   
   model.add(Bidirectional(LSTM(hidden_size, stateful=False), input_shape=(1, X.shape[2])))
   model.add(Dropout(0.5))
   model.add(Dense(19, activation='sigmoid'))
   model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])  
   history = model.fit(X, y, epochs=epochs, verbose=2, batch_size=20, callbacks=[tbCallBack], shuffle=True, validation_split=0.2)
   """
   loss.append(history.history['loss'])
   plt.plot(history.history['loss'])
   plt.title('Bidirectional LSTM loss')
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend(['train', 'test'], loc='upper left')
   """
   model.summary()
   #plt.show()
   return history, model

def train_RNNLSTM_model(X, y, hidden_size, epochs): 
   loss = list()
   model = Sequential()
   print(X.shape)
   print(y.shape)
   
   model.add(SimpleRNN(hidden_size))
   model.add(Dropout(0.5))
   model.add(Dense(19, activation='relu'))
   model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])  
   history = model.fit(X, y, epochs=epochs, verbose=2, batch_size=20, callbacks=[tbCallBack], shuffle=True, validation_split=0.2)
   """
   loss.append(history.history['loss'])
   plt.plot(history.history['loss'])
   plt.title('Bidirectional LSTM loss')
   plt.ylabel('loss')
   plt.xlabel('epoch')
   plt.legend(['train', 'test'], loc='upper left')
   """


   model.summary()
   #plt.show()
   return history, model


   def ranking_precision(y, y_pred_probs):
   presicion = 0
   for test in range(len(y)):
      presicion += 1 - rankdata(y_pred_probs[test])[np.argmax(y[test])-1]/len(y_pred_probs[test])

   score = presicion/len(y_pred_probs)

   return score

def evaluate_dataset(X1, y1, X2, y2, function):

   test_size = 0.3

   X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train, y2_test = train_test_split(X1, y1, X2, y2, test_size=test_size)
   X1_test, X1_valid, y1_test, y1_valid, X2_test, X2_valid, y2_test, y2_valid = train_test_split(X1_test, y1_test, X2_test, y2_test, test_size=0.5)


   data1 = []
   data2 = []
   data1.append([X1_train, y1_train])
   data1.append([X1_valid, y1_valid])
   data1.append([X1_test, y1_test])

   data2.append([X2_train, y2_train])
   data2.append([X2_valid, y2_valid])
   data2.append([X2_test, y2_test])

   # size of the input for view 1 and view 2
   

   # the parameters for training the network
   
   # the regularization parameter of the network
   # seems necessary to avoid the gradient exploding especially when non-saturating activations are used
  

   # specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
   # if one option does not work for a network or dataset, try the other one


   # if a linear CCA should get applied on the learned features extracted from the networks
   # it does not affect the performance on noisy MNIST significantly

   early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=1, verbose=2, mode='auto')
   





   model_classifier = KerasClassifier(build_fn=function, epochs=50, batch_size=100,
    verbose=1, )


   my_scorer = make_scorer(cca_loss, greater_is_better=True)

   validator = GridSearchCV(model_classifier,
                         param_grid={# epochs is avail for tuning even when not
                                     # an argument to model building function
                                     #'input_size': [int(mfcc1_X_train.shape[2])],
                                     'input_size1': [X1_train.shape[2]],
                                     'input_size2': [X2_train.shape[1]],
                                     'learning_rate': [1e-3, 0.005, 0.009 ,  ],
                                     'reg_par':[1e-4, 1e-3, 1e-2], 
                                     'outdim_size': [10,20,30,40,50],
                                     'use_all_singular_values': [False],
                                     'neurons_lstm': [32 , 64, 128, 265],
                                     'neurons1': [1024,512,256,128,64,32],
                                     'neurons2': [1024,512,256,128,64,32],
                                     'neurons3': [1024,512,256,128,64,32],
                                     'dropout': [0.2,0.3,0.4,0.5,0.6,0.7],
                                     'validation_split': [0.1,0.2,0.3],
                                     'batch_size': [1000],
                                     'epochs': [ 30, 120, 320],
                                     'activation_model1': ['sigmoid', 'relu','softmax' ],
                                     'activation_model2': ['sigmoid', 'relu','softmax' ],
                                     'activation_model3': ['sigmoid', 'relu','softmax' ],
                                     'dense_size': [8,16,32]
                                     },
                         scoring=my_scorer,
                         n_jobs=-1)
   
   #validator.fit(X_train, y_train)

   #filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
   checkpointer = ModelCheckpoint(filepath="temp_weights.h5", verbose=1, save_best_only=True, save_weights_only=True)
   validator.fit([X1_train, X2_train], y2_train)






   print('The parameters of the best model are: ')
   print(validator.best_params_)

   # validator.best_estimator_ returns sklearn-wrapped version of best model.
   # validator.best_estimator_.model returns the (unwrapped) keras model
   best_model = validator.best_estimator_.model
   best_model.summary()

   metric_names = best_model.metrics_names
   metric_values = best_model.evaluate(X_test, y_test)
   

   for metric, value in zip(metric_names, metric_values):
       print(metric, ': ', value)


   #plot_results(hist, epoch_num)

   new_data = test_dcca(best_model, data1, data2, outdim_size, apply_linear_cca)
   
   # Training and testing of SVM with linear kernel on the view 1 with new features
   [test_acc, valid_acc] = svm_classify(new_data, C=0.01)
   print("Accuracy on view 1 (validation data) is:", valid_acc * 100.0)
   print("Accuracy on view 1 (test data) is:", test_acc*100.0)



def train_dcca(X1, y1, X2, y2, epoch_num, batch_size=200):
   """
   This method is deprecated.
   The library used showed error on DCCA implementations and was substituted.
   Some techniques are used in the adaptations of the next tried implementation
   """
   test_size = 0.4  

   X1_train, X1_test, y1_train, y1_test, X2_train, X2_test, y2_train, y2_test = train_test_split(X1, y1, X2, y2, test_size=test_size)
   X1_test, X1_valid, y1_test, y1_valid, X2_test, X2_valid, y2_test, y2_valid = train_test_split(X1_test, y1_test, X2_test, y2_test, test_size=0.5)

   input_shape1= X1_train.shape[1]
   input_shape2= X2_train.shape[1]
   expert_data = []
   new_data = []

   input1 = Input(shape=(input_shape2, ), name='audios')
   input2 = Input(shape=(input_shape2, ), name='lyrics')


   activation_model = 'sigmoid'
   dense1_1 = Dense(512, activation=activation_model, name='view_1_1')(input1)
   dense1_2 = Dense(512, activation=activation_model, name='view_1_2')(dense1_1)
   dense1_3 = Dense(512, activation=activation_model,  name='view_1_3')(dense1_2)
   output1 = Dense(10, activation='linear', name='view_1_4')(dense1_3)

   dense2_1 = Dense(512, activation=activation_model,  name='view_2_1')(input2)
   dense2_2 = Dense(512, activation=activation_model,  name='view_2_2')(dense2_1)
   dense2_3 = Dense(512, activation=activation_model, name='view_2_3')(dense2_2)
   output2 = Dense(10, activation='linear', name='view_2_4')(dense2_3)

   shared_layer = concatenate([output1, output2], name='shared_layer')

   cca_layer = CCA(1, name='cca_layer')(shared_layer)

   model = Model(inputs=[input1, input2], outputs=cca_layer)
   model.compile(optimizer='rmsprop', loss=constant_loss, metrics=[mean_pred])
   hist = model.fit([X1_train, X2_train], np.zeros(len(X1_train)), batch_size=batch_size, epochs=epoch_num,
    shuffle=True, verbose=1,validation_data=([X1_valid, X2_valid], np.zeros(len(X2_valid))))
   
   current_dcca = Model(input=model.input, output=model.get_layer(name='shared_layer').output)
   train_pred = current_dcca.predict([X1_train, X2_train])
   valid_pred = current_dcca.predict([X1_valid, X2_valid])
   test_pred = current_dcca.predict([X1_test, X2_train])

   
   half = int(train_pred.shape[1] / 2)
   
   expert_data.append([train_pred[:, :half], train_pred[:, half:]])
   expert_data.append([valid_pred[:, :half], valid_pred[:, half:]])
   expert_data.append([test_pred[:, :half], test_pred[:, half:]])

   new_data.append([expert_data[0][0], expert_data[0][1], y1_train])
   new_data.append([expert_data[1][0], expert_data[1][1], y1_valid])
   new_data.append([expert_data[2][0], expert_data[2][1], y1_test])
   
   [test_acc, valid_acc] = svm_classify(new_data, C=0.01)
   print("current accuracy on view 1 (validation data) is:", valid_acc * 100.0)
   print("current accuracy on view 1 (test data) is:", test_acc * 100.0)

   print("training ended!")



def train_CNN_model(model, data1, data2, epoch_num, batch_size, label):
    """
    trains the model
    # Arguments
        data1 and data2: the train, validation, and test data for view 1 and view 2 respectively. data should be packed
        like ((X for train, Y for train), (X for validation, Y for validation), (X for test, Y for test))
        epoch_num: number of epochs to train the model
        batch_size: the size of batches
    # Returns
        the trained model
    """

    # Unpacking the data
    train_set_x1, train_set_y1 = data1[0]
    valid_set_x1, valid_set_y1 = data1[1]
    test_set_x1, test_set_y1 = data1[2]

    train_set_x2, train_set_y2 = data2[0]
    valid_set_x2, valid_set_y2 = data2[1]
    test_set_x2, test_set_y2 = data2[2]

    #train_set_x1_reshaped = train_set_x1.reshape(train_set_x1.shape[0], train_set_x1.shape[1], train_set_x1.shape[2], 1)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=6, verbose=2, mode='auto')
  
    filename = label +"_"+ "temp_cnn_weights.hdf5"
    checkpointer = ModelCheckpoint(filepath=filename, verbose=1, save_best_only=True, save_weights_only=True)
    # used dummy Y because labels are not used in the loss function

    hist = model.fit([train_set_x1], train_set_y1,batch_size=batch_size, epochs=epoch_num, shuffle=True,validation_data=(valid_set_x1, valid_set_y1),callbacks=[early_stopping, checkpointer])
    model.load_weights(filename)

    results = model.evaluate([test_set_x1], test_set_y1, batch_size=batch_size, verbose=1)

    print('loss on test data: ', results)

    results = model.evaluate([valid_set_x1], valid_set_y1, batch_size=batch_size, verbose=1)
    print('loss on validation data: ', results)


    

    return model, hist