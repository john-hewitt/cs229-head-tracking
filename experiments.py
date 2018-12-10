import random

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold
import sklearn

import util
import models
import experiments
from constants import test_participants
import matplotlib.pyplot as plt
from scipy.stats import mode
#import seaborn as sns
#sns.set(style="darkgrid")

def run_experiment(experiment_name, args):
  '''
  Uses python eval() to explicily evaluate the function named as experiment_name,
  with the args provided in args.
  '''
  experiment_function = eval(experiment_name)
  print('Running experiment: {}'.format(experiment_name))
  experiment_function(args)

def gad7_kfold_10(args):
  errs = [gad7_kfold(args) for x in range(10)]

def gad7_kfold_classification(args):
    '''
        Trains/evaluates models to predict the GAD7 variable from head movement data.
        Performs k-fold cross-validation.
        
        param args: a dictionary of arguments to be used in running this experiment.
    '''
    tracking_data = '../data/Tracking/'
    part_data = '../data/participant_data.csv'

    # load usable (pid, mo) pairs, and make sure to remove test set
    pid_mos_sg = util.which_parts_have_score(part_data, util.gad7)
    pid_mos_t = util.which_parts_have_tracking_data(tracking_data)
    pid_mos_both = list(set(pid_mos_sg) & set(pid_mos_t))
    pid_mos_use = list(filter(lambda pm : pm[0].upper() not in test_participants, pid_mos_both))
    #pid_mos_use = list(filter(lambda pm: pm[1] == 0, pid_mos_use))
    print('Loaded {} (pid, mo) pairs with both tracking data and GAD7 scores.'.format(len(pid_mos_both)))
    print('Removed {} (pid, mo) test set pairs to leave {} total to train with.'.format(len(pid_mos_both) - len(pid_mos_use), len(pid_mos_use)))

    X_train_dev = util.compute_fvecs_for_parts(pid_mos_use, args.featurization)#[:,1:10]
    scores = util.load_scores(part_data, pid_mos_use, util.gad7)
    y_train_dev = np.array(scores) >= 10

    kf = KFold(n_splits=118, random_state=1)
    kf_train_avg_errors = []
    kf_val_errors = []
    predictions = []
    for train_index, test_index in kf.split(X_train_dev):
      X_train, X_dev = X_train_dev[train_index], X_train_dev[test_index]
      y_train, y_dev = y_train_dev[train_index], y_train_dev[test_index]

      # Train model
      gad_model = models.ClassificationGAD7Model()
      gad_model.fit(X_train, y_train)

      y_train_predict = gad_model.predict(X_train) > .5
      y_train_error = np.mean(np.abs(y_train ^ y_train_predict))
      kf_train_avg_errors.append(y_train_error)

      # Predict on the held-out example
      y_predict = gad_model.predict(X_dev)
      predictions.append(y_predict[0])
      y_val_error = np.mean(np.abs(y_predict ^ y_dev))
      kf_val_errors.append(y_val_error)
      #print("\nModel predictions: {} \nTrue labels:       {} \n".format(y_predict, y_hold_out))
    #print("Avg GAD7 train error: {}".format(np.mean(kf_train_avg_errors)))
    print()
    print("GAD7 F1: {}".format(sklearn.metrics.f1_score(y_train_dev, predictions)))
    print("GAD7 prec: {}".format(sklearn.metrics.precision_score(y_train_dev, predictions)))
    print("GAD7 recall: {}".format(sklearn.metrics.recall_score(y_train_dev, predictions)))
    always_predict_true = [1 for x in y_train_dev]
    #print(y_train_dev)
    #print(predictions)
    #print("Always predict false: {}".format(np.mean(np.abs(y_train_dev ^ mode(y_train_dev)[0]))))
    print()
    print("Always predict false f1: {}".format(sklearn.metrics.f1_score(y_train_dev, always_predict_true)))
    print("Always predict false prec: {}".format(sklearn.metrics.precision_score(y_train_dev, always_predict_true)))
    print("Always predict false recall: {}".format(sklearn.metrics.recall_score(y_train_dev, always_predict_true)))
    random_predict = np.random.randint(0,2,y_train_dev.size)
    print()
    print("Predict randomly f1: {}".format(sklearn.metrics.f1_score(y_train_dev, random_predict)))
    print("Predict randomly prec: {}".format(sklearn.metrics.precision_score(y_train_dev, random_predict)))
    print("Predict randomly recall: {}".format(sklearn.metrics.recall_score(y_train_dev, random_predict)))
    print()
    print()
    random_20_percent = np.random.uniform(0,1, size=y_train_dev.size) <= .2373
    print("Predict random 20% have anxiety f1: {}".format(sklearn.metrics.f1_score(y_train_dev, random_20_percent)))
    print("Predict random 20% have anxiety prec: {}".format(sklearn.metrics.precision_score(y_train_dev, random_20_percent)))
    print("Predict random 20% have anxiety recall: {}".format(sklearn.metrics.recall_score(y_train_dev, random_20_percent)))
    print()
    print(np.mean(y_train_dev))
    #print("Avg GAD7 kfold val: {}".format(np.mean(kf_val_errors)))
    return np.mean(kf_val_errors)

def gad7_kfold(args):
    '''
        Trains/evaluates models to predict the GAD7 variable from head movement data.
        Performs k-fold cross-validation.
        
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

    X_train_dev = util.compute_fvecs_for_parts(pid_mos_use)[:,1:2]
    scores = util.load_scores(part_data, pid_mos_use, util.gad7)
    y_train_dev = np.array(scores) 

    kf = KFold(n_splits=118, random_state=1)
    kf_train_avg_errors = []
    kf_val_errors = []
    for train_index, test_index in kf.split(X_train_dev):
      X_train, X_dev = X_train_dev[train_index], X_train_dev[test_index]
      y_train, y_dev = y_train_dev[train_index], y_train_dev[test_index]
      print(X_dev, y_dev)

      # Train model
      gad_model = models.RegressionGAD7Model()
      gad_model.fit(X_train, y_train)

      y_train_predict = gad_model.predict(X_train)
      y_train_error = np.square(y_train - y_train_predict)
      kf_train_avg_errors.extend(y_train_error)

      # Predict on the held-out example
      y_predict = gad_model.predict(X_dev)
      y_val_error = np.square(y_predict - y_dev)
      kf_val_errors.extend(y_val_error)
      #print("\nModel predictions: {} \nTrue labels:       {} \n".format(y_predict, y_hold_out))
    print("Avg GAD7 train error: {}".format(np.mean(kf_train_avg_errors)))
    print("Avg GAD7 kfold val: {}".format(np.mean(kf_val_errors)))
    return np.mean(kf_val_errors)

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
    X_train_dev = util.compute_fvecs_for_parts(pid_mos_use)[:,0:2]
    scores = util.load_scores(part_data, pid_mos_use, util.gad7)
    y_train_dev = np.array(scores) 

    # Run hold-one-out evaluation on the train_dev set.
    hoo_train_avg_errors = []
    hoo_val_errors = []
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

      y_train_predict = gad_model.predict(X_hold_one_out_train)
      y_train_error = np.mean(np.square(y_hold_one_out_train - y_train_predict))
      hoo_train_avg_errors.append(y_train_error)

      # Predict on the held-out example
      y_predict = gad_model.predict(X_hold_out)
      print(y_predict)
      y_val_error = np.square(y_predict - y_hold_out)
      hoo_val_errors.append(y_val_error)
      #print("\nModel predictions: {} \nTrue labels:       {} \n".format(y_predict, y_hold_out))
    print("Avg GAD7 train error: {}".format(np.mean(hoo_train_avg_errors)))
    print("Avg GAD7 hold-one-out val: {}".format(np.mean(hoo_val_errors)))



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

def run_prelim_data_viz_exploration(args):
    '''
    Runs a bunch of visualization stuff about the GAD7 and SCL20 variables.
    '''
    print('Printing preliminary data exploration graphs')
    tracking_data = '../data/Tracking/'
    part_data = '../data/participant_data.csv'
    
    # load features and labels
    # load usable (pid, mo) pairs, and make sure to remove test set
    pid_mos_sg = util.which_parts_have_score(part_data, util.gad7)
    pid_mos_t = util.which_parts_have_tracking_data(tracking_data)
    pid_mos_both = list(set(pid_mos_sg) & set(pid_mos_t))
    pid_mos_use = list(filter(lambda pm : pm[0].upper() not in test_participants, pid_mos_both))
    print('Loaded {} (pid, mo) pairs with both tracking data and GAD7 scores.'.format(len(pid_mos_both)))
    print('Removed {} (pid, mo) test set pairs to leave {} total to train with.'.format(len(pid_mos_both) - len(pid_mos_use), len(pid_mos_use)))

    X_train_dev = util.compute_fvecs_for_parts(pid_mos_use)
    scores = util.load_scores(part_data, pid_mos_use, util.gad7)
    y_train_dev = np.array(scores) 

    X_train_dev_avgs = np.mean(X_train_dev, 0)
    X_train_dev_vars = np.var(X_train_dev, 0)
    X_train_dev = (X_train_dev - X_train_dev_avgs) / X_train_dev_vars
    sign = np.sign(X_train_dev)
    #X_train_dev = np.sqrt(np.sqrt(np.sqrt(np.abs(X_train_dev)))) * sign

    # Construct a histogram of the GAD7 data.
    fig1, ax1 = plt.subplots()
    ax1.hist(y_train_dev)
    ax1.set_title('GAD7 Train/Dev Data Distribution')
    ax1.set_xlabel('GAD7 Value')
    ax1.set_ylabel('# of partipants')
    fig1.tight_layout()
    fig1.savefig('gad7_hist.png', dpi=300)

    X_train_high_gad7_avg = np.mean(X_train_dev[y_train_dev >= 8], 0)
    X_train_low_gad7_avg = np.mean(X_train_dev[y_train_dev < 8], 0)
    fig1, ax1 = plt.subplots()
    ax1.set_title('Avg Feature Values for High- and Low-GAD7 Participants')
    ax1.scatter(range(len(X_train_high_gad7_avg)), X_train_high_gad7_avg, label='High-GAD7 participants')
    ax1.scatter(range(len(X_train_high_gad7_avg)), X_train_low_gad7_avg, label='Low-GAD7 participants')
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Feature Avg')
    fig1.tight_layout()
    ax1.legend()
    fig1.savefig('gad7_high_low.png', dpi=300)

    #y_train_dev = util.SCL20_labels(train_dev_participants, scores)
    #fig1, ax1 = plt.subplots()
    #ax1.hist(y_train_dev)
    #ax1.set_title('SCL20 Train/Dev Data Distribution')
    #ax1.set_xlabel('SCL20 Value')
    #ax1.set_ylabel('# of partipants')
    #fig1.tight_layout()
    #fig1.savefig('scl_hist.png', dpi=300)

    # Construct graphs of one single high-anxiety and one single low-anxiety participant.
    high_path = '../data/Tracking/TRACKING_MV08176R.TXT'
    high_val = 15

    low_path  = '../data/Tracking/TRACKING_MV0764712MOR.TXT'
    low_val = 2

    high_lines = [x.split('\t') for x in open(high_path)]
    high_lines = [(x[0], eval(x[1]), eval(x[2])) for x in high_lines]
    low_lines = [x.split('\t') for x in open(low_path)]
    low_lines = [(x[0], eval(x[1]), eval(x[2])) for x in low_lines]

    fig, axes = plt.subplots(3,2, figsize=(15,6))

    channel_names = ['Left Channel', 'Right Channel']
    axis_names = ['Pitch', 'Roll', 'Yaw']

    for axis_index in [0,1,2]:
      for channel_index in [0,1]:
        axes[axis_index][channel_index].scatter(range(len(high_lines)), [x[channel_index+1][axis_index] for x in high_lines], label='High-Anxiety Participant, GAD7=15')
        axes[axis_index][channel_index].scatter(range(len(low_lines)), [x[channel_index+1][axis_index] for x in low_lines], label='Low-Anxiety Participant, GAD7=2')
        axes[axis_index][channel_index].set_title(channel_names[channel_index] + ' ' + axis_names[axis_index])
    plt.tight_layout()
    plt.suptitle('Raw Head Movement Data for Sample High-GAD7 and Low-GAD7 Participants')
    plt.legend()
    plt.subplots_adjust(top=0.88)
    plt.savefig('head_tracking_example.png', dpi=300)


def gad7_single_predictor_analysis(args):
    '''
    Trains/evaluates models to predict the GAD7 variable from head movement data.
    
    Specifically takes a single variable for each linear regression for prediction,
    attempting to roughly sort predictors by predictiveness.
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
    sklearn.preprocessing.normalize(X_train_dev)
    #X_train_dev_avgs = np.mean(X_train_dev, 0)
    #X_train_dev_vars = np.var(X_train_dev, 0)
    #X_train_dev = (X_train_dev - X_train_dev_avgs) / X_train_dev_vars

    scores = util.load_scores(part_data, pid_mos_use, util.gad7)
    y_train_dev = np.array(scores) 

    # Run hold-one-out evaluation on the train_dev set.
    hoo_train_avg_errors = []
    hoo_val_errors = []
    feature_errors = []
    #for movie_index in range(5):
    for feature_index in range(X_train_dev.shape[1]):
        #start_index = movie_index * 24
        #end_index = (movie_index+1)*24
        for hold_out_index, _ in enumerate(X_train_dev):
            X_hold_out = X_train_dev[hold_out_index].reshape(1, -1)[:,feature_index].reshape(-1,1)#[:,start_index:end_index]
            #X_hold_out = X_train_dev[hold_out_index].reshape(1, -1)#[:,start_index:end_index]
            y_hold_out = y_train_dev[hold_out_index]

            X_hold_one_out_train = np.array(X_train_dev[:hold_out_index].tolist() 
                + X_train_dev[hold_out_index+1:].tolist())[:,feature_index].reshape(-1,1)#[:,start_index:end_index] # Take out the HOO example
               #+ X_train_dev[hold_out_index+1:].tolist())#[:,start_index:end_index] # Take out the HOO example
            y_hold_one_out_train = np.array(y_train_dev[:hold_out_index].tolist() 
                + y_train_dev[hold_out_index+1:].tolist()) # Take out the HOO example

            # Train model
            gad_model = models.RegressionGAD7Model()
            gad_model.fit(X_hold_one_out_train, y_hold_one_out_train)

            y_train_predict = gad_model.predict(X_hold_one_out_train)
            y_train_error = np.mean(np.square(y_hold_one_out_train - y_train_predict))
            hoo_train_avg_errors.append(y_train_error)

            # Predict on the held-out example
            y_predict = gad_model.predict(X_hold_out)
            y_val_error = np.square(y_predict - y_hold_out)
            hoo_val_errors.append(y_val_error)
            #print("\nModel predictions: {} \nTrue labels:       {} \n".format(y_predict, y_hold_out))
        print("Avg GAD7 train error for feature {}: {}".format(feature_index, np.mean(hoo_train_avg_errors)))
        print("Avg GAD7 hold-one-out val for feture {}: {}".format(feature_index, np.mean(hoo_val_errors)))
        #print("Avg GAD7 train error for feature {}: {}".format(movie_index, np.mean(hoo_train_avg_errors)))
        #print("Avg GAD7 hold-one-out val for feture {}: {}".format(movie_index, np.mean(hoo_val_errors)))
        feature_errors.append(np.mean(hoo_val_errors))
    for index, error in sorted(list(enumerate(feature_errors)), key=lambda x: x[1]):
      print(index, error)

