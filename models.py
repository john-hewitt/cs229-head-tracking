import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import util
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
        # do this all outside of fit? pass in via X, y ... and then don't have to return X,y
        '''       tracking_data = '../data/test.txt'
        part_data = '../data/participant_data.csv'
        
        # get PARTS 
        parts = ['LA13272', 'LA14016', 'MV00962', 'MV01113', 'MV01950', 'MV07296', 'MV07303','MV07647','MV08032','MV09122', 'MV09305', 'MV09441', 'MV09586','MV11133','MV11150', 'MV11202', 'PA22014', 'PA22544','PA22561','PA22728','PA23284', 'PA23955','PA24326', 'PA24859','PA24876','PA25084','PA25119',  'PA25306','PA26203','PA26376', 'PA26623', 'PA27784','PA27793','PA27962','PA30895', 'PA30677', 'PA30862', 'PA30895', 'SU30734', 'SU30816','SU33550','SU35282']
        
        # load features
        train_matrix = util.compute_fvecs_for_parts(parts)
        
        # load labels
        score_dict = util.load_participant_scores(part_data)
        
        # get gad7 labels
        gad_labels = util.GAD7_labels(parts, score_dict)
        ''' 
        # Need to make sure we have model predicitions in the appropriate range [0 - ~25?]
        # maybe it would be better to do this as categorical multinomial? buckets of how given in data dictionary
        # Measure of anxiety 1-4 below threshold, 5-9 mild, 10-14 moderate, 15+ severe

        # Train linear regression model with lasso regularization 
        # lasso? ridge? ridgeCV? what should we use!  figure that out using eval set?
        clf = linear_model.Lasso(alpha = 0.0001, max_iter=1e5)
        
        clf.fit(X, y)

        # Get theta
        theta_coeff = clf.coef_

        # Save the fit model + coeffs
        self.theta_coeff = theta_coeff
        self.clf = clf
        self.trained = True

        return theta_coeff


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
