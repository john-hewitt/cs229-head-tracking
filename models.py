import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import util
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
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
        clf = linear_model.Lasso(alpha = 0.0001, max_iter=1e8)
    #    clf = tree.DecisionTreeRegressor(min_samples_leaf=8)
        #clf = AdaBoostRegressor(max_depth = 4,  n_estimators = 100)
        clf.fit(X, y)

        self.clf = clf
        self.trained = True

        y_pred = clf.predict(X)

        return y_pred
   #      return theta_coeff


    def predict(self, x):        
        """
        """
        return np.around(self.clf.predict(x), decimals = 0)
    




class LearnedEnsembleClassificationGAD7Model(Model):
  '''
  Class containing functions for fitting/predicting a classification model,
  where each prediction is a learned ensemble of a set of models specified
  at construction.
  '''

  def __init__(self, model_retriever, weights, threshold):
    self.model_retriever = model_retriever
    self.weights = weights
    self.threshold = threshold

  def fit(self, X, y):
    '''
    Fits all models given by the model_retriever specified
    (models should be re-retrieved each time as otherwise they'll get stale)
    '''
    self.models = []
    for model in self.model_retriever():
      self.models.append(model)
      model.fit(X,y)

  def predict(self, x):
    '''
    Prediction through learned ensemble and cutoff
    '''
    predictions = np.array([model.predict(x) for model in self.models])
    weighted_predictions = np.multiply(np.expand_dims(self.weights,1), predictions)
    decision = np.sum(weighted_predictions, 0) > self.threshold
    return decision



class ClassificationGAD7Model(Model):
    '''
    Class containing functions for fitting/predicting a classification model,
    specifically for the GAD7 variable but this is somewhat arbitrary.
    '''

    def fit(self, X, y):
        """ Fits a classification model to predict major depressive disorder
        based on N participants' head movement data.
        
        Returns a tuple (X, Y, Theta) where X is a N-by-M numpy array,

        In particular, at this point fits _three_ different types of classifiers
        for ensembling.
        """
        
        # Train logistic regression model                   
        self.logreg_clf = LogisticRegression(max_iter=1000000000) # lbfgs
        self.logreg_clf.fit(X, y)
        self.multinom_clf = MultinomialNB() # lbfgs
        self.multinom_clf.fit(X, y)
        self.tree_clf = tree.DecisionTreeClassifier()
        self.tree_clf.fit(X,y)
        #clf = RandomForestClassifier(n_estimators=3, bootstrap=False)
        self.trained = True

        #return theta_coeff


    def predict(self, x):
        """
        Prediction through ensemble majority vote. 
        """
        pred_1 = self.logreg_clf.predict(x)
        pred_2 = self.multinom_clf.predict(x)
        pred_3 = self.tree_clf.predict(x)
        #pred = (pred_1 ^ pred_2 ^ pred_3)
        #print(pred_1*1+pred_2*1+pred_3*1)
        pred = (pred_3*1 + pred_2*1 + pred_1*1)/2 > .5
        #pred = pred_3
        #pred = (pred_1*1.1 + pred_2*1.1 + pred_3*2)/(2+1.1+1.1) > .5
        #pred = pred_3

        #print()
        #print(pred_1, pred_2, pred_3)
        #print(pred)
        #input()
        
        return pred
        #return self.clf.predict(x)
