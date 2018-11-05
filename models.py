import util
from sklearn import linear_model
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

    def fit(self):
        """ Fits a regression model to predict GAD7 anxiety scores based on 
        N participants' head movement data.
        
        Returns a tuple (X, Y, Theta) where X is a N-by-120 numpy array,
        Y is a N-by-1, and Theta is 120-by-1.

        Note: we can change this architecture up - I'm open to suggestions :)
        """
        tracking_data = '../data/test.txt'
        part_data = '../data/participant_data.csv'
        
        # get PARTS 
        # parts = 
        # for now..
        parts = ['LA13272', 'LA14016', 'MV00962', 'MV01113', 'MV01950', 'MV07296', 'MV07303','MV07647','MV08032','MV09122', 'MV09305', 'MV09441', 'MV09586','MV11133','MV11150', 'MV11202', 'PA22014', 'PA22544','PA22561','PA22728','PA23284', 'PA23955','PA24326', 'PA24859','PA24876','PA25084','PA25119',  'PA25306','PA26203','PA26376', 'PA26623' 'PA27784','PA27793','PA27962','PA30895', 'PA30677', 'PA30862', 'PA30895', 'SU30734', 'SU30816','SU33550','SU35282']

        # load features
        train_matrix = util.compute_fvecs_for_parts(parts)

        # load labels
        score_dict = util.load_participant_scores(part_data)
        
        # get gad7 labels
        gad_labels = util.GAD7_labels(score_dict)

        # train linear regression model with lasso regularization
        clf = linear_model.Lasso(alpha = 0.1)
        clf.fit(train_matrix, gad_labels)

        # get theta
        theta_coeff = clf.coef_

        # Save the fit model + coeffs
        self.theta_coeff = theta_coeff
        self.clf = clf
        return (X, y, theta_coeff)


class ClassificationSLC20Model(Model):
    '''
    Class containing functions for fitting/predicting a classification model,
    specifically for the GAD7 variable but this is somewhat arbitrary.
    '''

    def fit():
        """ Fits a classification model to predict major depressive disorder
        based on N participants' head movement data.
        
        Returns a tuple (X, Y, Theta) where X is a N-by-120 numpy array,
        Y is a N-by-1, and Theta is 120-by-1.

        Note: we can change this architecture up - I'm open to suggestions :)
        """
        tracking_data = '../data/test.txt'
        part_data = '../data/participant_data.csv'

        # get PARTS                                                                                           
        # parts =                                                                                                                                                                                               
        # load features                                                                          
        train_matrix = util.compute_fvecs_for_parts(parts)

        # load labels                                                                                     
        score_dict = util.load_participant_scores(part_data)

        # get scl20 labels  - 0 or 1                                                        
        scl_labels = util.SCL20_labels(parts, score_dict)
    

