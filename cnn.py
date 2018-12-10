from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import AveragePooling1D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
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

    def  __init__(self):
        self.trained =  False
        self.X_train = None
        self.y_train = None

    def fit(self):
        '''            
        Fits a CNN model to learn to classify anxiety based on 118 participants head movement data broken up into 
        2179 timesteps across 6 channels 
        '''
        
        # 32, 10, .5 , :300  -> : p = 0.214,   R = 0.429,  F1 = 0.2876
        # 32, 20, .5 , :500  -> 

        # X is matrix of all training examples, of shape (2179 timesteps x 6 channels x 118 examples)
        # reshape to the required  keras format of (batchsize, steps, channels) = (118, 2179, 6) 
        x_train = self.X_train
        x_train = np.transpose(x_train, (2,0,1))
        x_train_trunc = x_train[:,:500,:]
        batch_size = self.y_train.shape[0]
        y_preds = np.zeros(batch_size)

        for i in range(0, batch_size):
            print("\n Iteration {}: \n ".format(i))
        
            model = Sequential()
        
            conv = Conv1D(32, (20), input_shape = (500, 6)) 
            model.add(conv) 
            pool = AveragePooling1D(pool_size = 480)
            model.add(pool)  
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(32, activation = 'relu'))
            model.add(Dense(1, activation= 'sigmoid'))
            
            sgd = optimizers.SGD(lr = 1)
            model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

            x_held_out = (x_train_trunc[i,:,:])
            x_held_out = x_held_out.reshape(1, x_held_out.shape[0],  x_held_out.shape[1])
            y_held_out = self.y_train[i]
            x_train_hoo = np.delete(x_train_trunc, i, axis = 0)
            y_train_hoo = np.delete(self.y_train, i, axis = 0)
            model.fit(x_train_hoo, y_train_hoo, epochs = 400, batch_size = 60)
            # preds =  model.predict(x_train_hoo, batch_size = 117)
            y_preds[i] = model.predict(x_held_out, batch_size = 1, verbose = 1)   # predict on held out ex
        
        # convert to labels
        y_preds = (y_preds > 0.5).astype(int)

        #compute metrics
        precision = precision_score(y_preds, self.y_train) 
        recall = recall_score(y_preds, self.y_train) 
        f1 = f1_score(y_preds, self.y_train) 

        print(" -------------------- ")
        print("y_preds: {} \n y_true: {} \n".format(y_preds, self.y_train))
        print(" Precision: {} \n Recall: {} \n F1: {} \n ".format(precision, recall, f1)) 
        print(" -------------------- ")
        
        
        self.model = model
        self.trained = True

    def predict(self, X, y):
        '''  
        Uses trained model parameters to predict values for unseen data.
        Raises ValueError if the model has not yet been trained..  
        '''
        if not self.trained:
            raise ValueError("This model has not been trained yet")

        batch_size = X.shape[0]
        pred = self.model.predict(X, batch_size)
        self.pred = pred

        return pred


    def load_data(self, tracking_data, part_data):
        ''' 
        Loads raw head movement data and participant data from tracking_data file and part_data file
        '''

        tracking_data = '../data/Tracking/'
        part_data= '../data/participant_data.csv'
        
        # load features and labels   
        # load usable (pid, mo) pairs, and make sure to remove test set                              
        pid_mos_sg = util.which_parts_have_score(part_data, util.gad7)
        pid_mos_t = util.which_parts_have_tracking_data(tracking_data)
        pid_mos_both = list(set(pid_mos_sg) & set(pid_mos_t))
        pid_mos_use = list(filter(lambda pm : pm[0].upper() not in test_participants, pid_mos_both))
        print('Loaded {} (pid, mo) pairs with both tracking data and GAD7 scores.'.format(len(pid_mos_both)))
        print('Removed {} (pid, mo) test set pairs to leave {} total to train with.'.format(len(pid_mos_both) - len(pid_mos_use), len(pid_mos_use)))
        
        num_train = len(pid_mos_use)
        max_timesteps = 0
        first = True
        for pid, mo in pid_mos_use:
            tfiles = [util.tracking_file(pid, mo, exp) for exp in exps]   
            expvecs = [util.load_raw_hm_data(tfile) for tfile in tfiles]  # 5 sets of timestep values for each experience type for this pid.mo pair 
            timesteps =  [ x.shape[0] for x in expvecs ]
            fullvec = np.concatenate(expvecs, axis=0)
            padded_full = np.pad(fullvec, ((0, max_dim - fullvec.shape[0]), (0,0)), 'constant')
            if first == True:
                X_train_dev = padded_full.reshape((padded_full.shape[0], padded_full.shape[1], 1))
                first = False
            else:
                X_train_dev= np.dstack((X_train_dev, padded_full))
                
    
        scores = util.load_scores(part_data, pid_mos_use, util.gad7)
        y_train_dev = np.array(scores) >= 10
        X_train_dev = X_train_dev/np.linalg.norm(X_train_dev, ord=np.inf, axis=0, keepdims=True)

        self.X_train = X_train_dev
        self.y_train = y_train_dev.astype(int)

        
