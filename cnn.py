from keras.models import Sequential, Model
from keras.layers import Activation, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, Dense, Dropout
from keras import optimizers
from keras import backend as K
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score

import util
import experiments
from constants import test_participants
import numpy as np
import sklearn

# globals                                                                                                                                 
mos = [0, 2, 6, 12]
exps = ['R', 'N1', 'N2', 'P1', 'P2']
max_dim = 2179

class CNN:
    ''' 
    Class containing functions for fitting/predicting a CNN model for the presence of anxiety
    based on the GAD7 anxiety score
    '''

    def  __init__(self):
        self.trained =  False
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def fit(self):
        '''            
        Fits a CNN model to learn to classify anxiety based on 118 participants head movement data broken up into 
        2179 timesteps across 6 channels 
        Uses parameters found in hyperparameter tuning via fit_hold_on_out to train on full batch
        '''
        
        # X is matrix of all training examples, of shape (2179 timesteps x 6 channels x 118 examples)
        # Reshape to the required format: (batchsize, steps, channels) = (118, 2179, 6) 
        x_train = self.X_train
        x_train = np.transpose(x_train, (2,0,1))
        x_train_trunc = x_train[:,:300,:]
        batch_size = self.y_train.shape[0]
        y_preds = np.zeros(batch_size)
        
        model = Sequential()

        conv = Conv1D(16, (10), input_shape = (300, 6)) 
        model.add(conv) 
        pool = AveragePooling1D(pool_size = 291)
        model.add(pool)  
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(32, activation = 'relu'))
        model.add(Dense(1, activation= 'sigmoid'))
            
        sgd = optimizers.SGD(lr = 1)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(x_train_trunc, self.y_train, epochs = 300, batch_size = 60)
        
        self.model = model
        self.trained = True
        
    def fit_hold_one_out():

        '''                                                                                                                                        
        Fits a CNN model via hold on out to learn to classify anxiety based on 118 participants head movement data broken up into                     
        2179 timesteps across 6 channels                                                                                                   
        Used to find set of hyperparameters for the CNN  that yields the best combination of precision, recall, and f1 scores
        '''                                                                                   
                                              
        # Reshape X ( 2179 timesteps x 6 channels x  118 ex) to the required  format of (batchsize, steps, channels) = (118, 2179, 6)            
        x_train = self.X_train
        x_train = np.transpose(x_train, (2,0,1))
        x_train_trunc = x_train[:,:300,:]
        batch_size = self.y_train.shape[0]
        y_preds = np.zeros(batch_size)

        for i in range(0, batch_size):                                                                                                                 
            print("\n Starting iteration {}: \n ".format(i))                                                                                                    
            model = Sequential()
                                                                                                                                                  
            conv = Conv1D(16, (10), input_shape = (500, 6))
            model.add(conv)
            pool = AveragePooling1D(pool_size = 481)
            model.add(pool)
            model.add(Activation('relu'))
            model.add(Dropout(0.3))
            model.add(Flatten())
            model.add(Dense(32, activation = 'relu'))
            model.add(Dense(1, activation= 'sigmoid'))

            sgd = optimizers.SGD(lr = 1)
            model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

            x_held_out = (x_train_trunc[i,:,:])  # on each iteration hold out the ith example                                                    
            x_held_out = x_held_out.reshape(1, x_held_out.shape[0],  x_held_out.shape[1])                                                
            y_held_out = self.y_train[i]                                                                                  
            x_train_hoo = np.delete(x_train_trunc, i, axis = 0)                                                                    
            y_train_hoo = np.delete(self.y_train, i, axis = 0)                                                                      
            model.fit(x_train_hoo, y_train_hoo, epochs = 1, batch_size = 60)                                                          
            y_preds[i] = model.predict(x_held_out, batch_size = 1, verbose = 1)   # predict on held out ex                                                                                                                                                                           
        y_preds = (y_preds > 0.5).astype(int)
  
        # Compute metrics                                                                                                   
        precision, recall, f1 = self.compute_metrics(y_preds, self.y_train)

        # Print summary                                                                                                                     
        print(" ------------------------------------ ")
        print("Prediction y values: {} \n True y values: {} \n".format(y_preds, self.y_train))
        print(" Precision: {} \n Recall: {} \n F1: {} \n ".format(precision, recall, f1))
        print(" ------------------------------------ ")

        self.model = model                                                                                                        
        self.trained = True    


    def predict(self):
        '''  
        Uses trained model parameters to predict values for unseen data via model.predict.
        For now uses test data stored in self.X_test and self.y_test.
        Raises ValueError if the model has not yet been trained..  
        Returns a numpy array of predictions 
        '''

        if not self.trained:
            raise ValueError("This model has not been trained yet")

        # Reshape test data into correct format of (batch_size, steps, channels) over same amount of timesteps  
        X_test =  np.transpose(self.X_test, (2,0,1))
        X_test = X_test[:,:300,:]
        batch_size = self.y_test.shape[0]
        y_preds = np.zeros(batch_size)
        
        y_preds = self.model.predict(X_test, batch_size)
        y_preds = (y_preds > 0.5).astype(int)
        self.pred = y_preds

        return y_preds


    def load_data(self, tracking_data, part_data):
        ''' 
        Loads raw head movement data and participant data from tracking_data file and part_data file
        Splits data into train/dev set and test set and saves X_train, y_train, X_test, and y_test
        '''

        tracking_data = '../data/Tracking/'
        part_data= '../data/participant_data.csv'
        
        # load features and labels   
        # load usable (pid, mo) pairs, and make sure to remove test set                              
        pid_mos_sg = util.which_parts_have_score(part_data, util.gad7)
        pid_mos_t = util.which_parts_have_tracking_data(tracking_data)
        pid_mos_both = list(set(pid_mos_sg) & set(pid_mos_t))
        pid_mos_use = list(filter(lambda pm : pm[0].upper() not in test_participants, pid_mos_both))
        pid_mos_test =  list(filter(lambda pm : pm[0].upper() in test_participants, pid_mos_both))

        num_train = len(pid_mos_use)
        num_test = len(pid_mos_test)

        print('Loaded {} (pid, mo) pairs with both tracking data and GAD7 scores.'.format(len(pid_mos_both)))
        print('Removed {} (pid, mo) test set pairs to leave {} total to train with.'.format(num_test, num_train))
        
        # Load train data
        first = True
        for pid, mo in pid_mos_use:
            tfiles = [util.tracking_file(pid, mo, exp) for exp in exps]   
            # 5 sets of timestep values for each experience type for this pid.mo pair  
            expvecs = [util.load_raw_hm_data(tfile) for tfile in tfiles]   
            timesteps =  [ x.shape[0] for x in expvecs ]
            fullvec = np.concatenate(expvecs, axis=0)
            padded_full = np.pad(fullvec, ((0, max_dim - fullvec.shape[0]), (0,0)), 'constant')
            if first == True:
                X_train_dev = padded_full.reshape((padded_full.shape[0], padded_full.shape[1], 1))
                first = False
            else:
                X_train_dev= np.dstack((X_train_dev, padded_full))
                
        scores_train = util.load_scores(part_data, pid_mos_use, util.gad7)
        y_train = np.array(scores_train) >= 10
        X_train_dev = X_train_dev/np.linalg.norm(X_train_dev, ord=np.inf, axis=0, keepdims=True)


        # Load test data
        first = True
        for pid, mo in pid_mos_test:
            tfiles = [util.tracking_file(pid, mo, exp) for exp in exps]
            expvecs = [util.load_raw_hm_data(tfile) for tfile in tfiles]                                                                            
            timesteps =  [ x.shape[0] for x in expvecs ]
            fullvec = np.concatenate(expvecs, axis=0)
            padded_full = np.pad(fullvec, ((0, max_dim - fullvec.shape[0]), (0,0)), 'constant')
            if first == True:
                X_test = padded_full.reshape((padded_full.shape[0], padded_full.shape[1], 1))
                first = False
            else:
                X_test = np.dstack((X_test, padded_full))

        scores_test = util.load_scores(part_data, pid_mos_test, util.gad7)
        y_test = np.array(scores_test) >= 10
        X_test = X_test/np.linalg.norm(X_test, ord=np.inf, axis=0, keepdims=True)


        self.X_train = X_train_dev
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test.astype(int)

        

    def compute_metrics(self, y_preds, y):
        ''' 
        Computes precision, recall,  and F1 score  metrics  for a given set 
        of predictions y_pred and true values y                                                                                           
        Returns  a tuple of (precision, recall,  f1)
        '''

        precision = precision_score(y_preds, y) 
        recall = recall_score(y_preds, y)                                                                      
        f1 = f1_score(y_preds, y)

        return (precision, recall, f1)
