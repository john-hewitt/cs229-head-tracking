import random

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import util
import models
import experiments
from constants import test_participants

def run_experiment(experiment_name, args):
  '''
  Uses python eval() to explicily evaluate the function named as experiment_name,
  with the args provided in args.
  '''
  experiment_function = eval(experiment_name)
  print('Running experiment: {}'.format(experiment_name))
  experiment_function(args)

def gad7_hold_one_out(args):
    '''
    Trains/evaluates models to predict the GAD7 variable from head movement data.
    
    param args: a dictionary of arguments to be used in running this experiment.
    '''
    tracking_data = '../data/Tracking/'
    part_data = '../data/participant_data.csv'

    # load usable (pid, mo) pairs, and make sure to remove test set
    pid_mos_sg = util.which_parts_have_score(part_data, util.gad7)
    pid_mos_t = util.which_parts_have_tracking_data(tracking_data)
    pid_mos_both = list(set(pid_mos_sg) & set(pid_mos_t))
    pid_mos_use = list(filter(lambda pm : pm[0].upper() not in test_participants, pid_mos_both))
    print('Loaded {} (pid, mo) pairs with both tracking data and GAD7 scores.'.format(len(pid_mos_both)))
    print('Removed {} (pid, mo) test set pairs to leave {} total to train with.'.format(len(pid_mos_both) - len(pid_mos_use), len(pid_mos_use)))

    # load features and labels
    X_train_dev = util.compute_fvecs_for_parts(pid_mos_use)
    scores = util.load_scores(part_data, pid_mos_use, util.gad7)
    y_train_dev = np.array(scores) 

    # Run hold-one-out evaluation on the train_dev set.
    for hold_out_index, _ in enumerate(X_train_dev):
      X_hold_out = X_train_dev[hold_out_index].reshape(1, -1)
      y_hold_out = y_train_dev[hold_out_index]

      X_hold_one_out_train = np.array(X_train_dev[:hold_out_index].tolist() 
          + X_train_dev[hold_out_index+1:].tolist()) # Take out the HOO example
      y_hold_one_out_train = np.array(y_train_dev[:hold_out_index].tolist() 
          + y_train_dev[hold_out_index+1:].tolist()) # Take out the HOO example

      # Train model
      gad_model = models.RegressionGAD7Model()
      gad_model.fit(X_hold_one_out_train, y_hold_one_out_train)

      # Predict on the held-out example
      y_predict = gad_model.predict(X_hold_out)
      print("\nModel predictions: {} \nTrue labels:       {} \n".format(y_predict, y_hold_out))


def run_slc20_experiment(args):
    '''
    Trains/evaluates models to predict the SLC20 variable from head movement data.

    param args: a dictionary of arguments to be used in running this experiment.
    '''
    tracking_data = '../data/test.txt'
    part_data = '../data/participant_data.csv'

    # load features and labels                                                                               
    X_train_dev = util.compute_fvecs_for_parts(train_dev_participants)
    scores = util.load_participant_scores(part_data)
    y_train_dev = util.SCL20_labels(train_dev_participants, scores)

    for hold_out_index, _ in enumerate(X_train_dev):
      X_hold_out = X_train_dev[hold_out_index].reshape(1, -1)
      y_hold_out = y_train_dev[hold_out_index]

      X_hold_one_out_train = np.array(X_train_dev[:hold_out_index].tolist() 
          + X_train_dev[hold_out_index+1:].tolist()) # Take out the HOO example
      y_hold_one_out_train = np.array(y_train_dev[:hold_out_index].tolist() 
          + y_train_dev[hold_out_index+1:].tolist()) # Take out the HOO example

      # Train the model
      slc_model = models.ClassificationSLC20Model()
      slc_model.fit(X_hold_one_out_train, y_hold_one_out_train)
      
      # Predict on test set and evaluate
      y_predict = slc_model.predict(X_hold_out)
      print("\nModel predictions: {}".format(y_predict))
      print("True labels:       {} \n".format(y_hold_out))
