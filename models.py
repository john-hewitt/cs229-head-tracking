import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import util
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import random

class Model:
    '''
    Abstract class to show what functions
    each model we implement needs to support
    '''

    def __init__(self):
        '''
        Sets flags for the model to aid in debugging
        '''
        self.trained = False

    def fit(self, *args):
        '''
        Trains model parameters and saves them as attributes of this class.

        Variable numbers of parameters; depends on the class.
        '''
        raise NotImplementedError

    def predict(self, *args):
        '''
        Uses trained model parameters to predict values for unseen data.
        Raises ValueError if the model has not yet been trained.
        Variable numbers of parameters; depends on the class.
        '''
        if not self.trained:
            raise ValueError("This model has not been trained yet")
        raise NotImplementedError


class RegressionGAD7Model(Model):
    '''
    Class containing functions for fitting/predicting a regression model,
    specifically for the GAD7 variable but this is somewhat arbitrary.
    '''

    def fit(self, X, y):
        """ Fits a regression model to predict GAD7 anxiety scores based on 
        N participants' head movement data.
        
        Returns a tuple (X, Y, Theta) where X is a N-by-120 numpy array,
        Y is a N-by-1, and Theta is 120-by-1.

        Note: we can change this architecture up - I'm open to suggestions :)
        """        

        # Need to make sure we have model predicitions in the appropriate range [0 - ~25?]
    #    #clf = linear_model.Lasso(alpha = 1, max_iter=1e8)
    #    clf = tree.DecisionTreeRegressor(min_samples_leaf=8)
        clf = AdaBoostRegressor(max_depth = 4,  n_estimators = 100)
        clf.fit(X, y)

        # Get theta
    #      theta_coeff = clf.coef_

        # Save the fit model + coeffs
    #    self.theta_coeff = theta_coeff
        self.clf = clf
        self.trained = True

        y_pred = clf.predict(X)

        return y_pred
   #      return theta_coeff


    def predict(self, x):        
        """
        """
        return np.around(self.clf.predict(x), decimals = 0)
    

class ClassificationSLC20Model(Model):
    '''
    Class containing functions for fitting/predicting a classification model,
    specifically for the GAD7 variable but this is somewhat arbitrary.
    '''

    def fit(self, X, y):
        """ Fits a classification model to predict major depressive disorder
        based on N participants' head movement data.
        
        Returns a tuple (X, Y, Theta) where X is a N-by-120 numpy array,
        Y is a N-by-1, and Theta is 120-by-1.

        Note: we can change this architecture up - I'm open to suggestions :)
        """
        
        # Train logistic regression model                   
        clf = LogisticRegressionCV(cv = 4, random_state=0, solver='newton-cg', penalty = 'l2') # lbfgs
        clf.fit(X, y)

        # get theta                                     
        theta_coeff = clf.coef_

        # Save the fit model + coeffs                                
        self.theta_coeff = theta_coeff
        self.clf = clf
        self.trained = True

        return theta_coeff


    def predict(self, x):
        """
        
        """
        return self.clf.predict(x)
