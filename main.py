from argparse import ArgumentParser
import random

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import util
import models

# for testing for now
train_dev_participants = ['LA13272', 'LA14016', 'MV00962', 'MV01113', 'MV01950', 'MV07296', 'MV07303','MV07647','MV08032','MV09122', 'MV09305', 'MV09441', 'MV09586','MV11133','MV11150', 'MV11202', 'PA22014', 'PA22544','PA22561','PA22728','PA23284', 'PA23955','PA24326', 'PA24859','PA24876','PA25084','PA25119',  'PA25306','PA26203','PA26376', 'PA26623', 'PA27784','PA27793','PA27962',]

test_participants = ['PA30677', 'PA30862', 'PA30895', 'SU30734', 'SU30816','SU33550','SU35282','PA30895']

# JOHN

def run_gad7_experiment(args):
    '''
    Trains/evaluates models to predict the GAD7 variable from head movement data.
    
    param args: a dictionary of arguments to be used in running this experiment.
    '''
    tracking_data = '../data/test.txt'
    part_data = '../data/participant_data.csv'
    
    # load features and labels
    X_train_dev = util.compute_fvecs_for_parts(train_dev_participants)
    scores = util.load_participant_scores(part_data)
    y_train_dev = util.GAD7_labels(train_dev_participants, scores)

    # Run hold-one-out evaluation on the train_dev set.
    for hold_out_index, _ in enumerate(X_train_dev):
      X_hold_out = X_train_dev[hold_out_index].reshape(1, -1)
      y_hold_out = y_train_dev[hold_out_index]
      #print(X_hold_out.shape)
      #print(y_hold_out.shape)

      X_hold_one_out_train = np.array(X_train_dev[:hold_out_index].tolist() 
          + X_train_dev[hold_out_index+1:].tolist()) # Take out the HOO example
      y_hold_one_out_train = np.array(y_train_dev[:hold_out_index].tolist() 
          + y_train_dev[hold_out_index+1:].tolist()) # Take out the HOO example
      #print(X_hold_one_out_train.shape)
      #print(y_hold_one_out_train.shape)

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
    X = util.compute_fvecs_for_parts(parts)
    scores = util.load_participant_scores(part_data)
    y = util.SCL20_labels(parts, scores)

    # Split data into train, eval, and test sets                                                                          
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    # Train the model
    slc_model = models.ClassificationSLC20Model()
    slc_model.fit(X_train, y_train)
    
    # Predict on test set and evaluate
    y_predict = slc_model.predict(X_test)
    print("\nModel predictions: {}".format(y_predict))
    print("True labels:       {} \n".format(y_test))

# SARAH / COOPER
    
# JOHN / all
def main(args):
    """ Fits models to predict mental health outcomes (anxiety, 
    depression) based on head movement data gathered during various
    virtual reality experiences.

    Analyzes these models' efficacy on a test set and (maybe) on 
    future (2 month, 6 month, 12 month) data.
    """
    
    print('Running GAD7 experiment...')
    run_gad7_experiment(args)
    print('GAD7 experiment finished.')
    print()
    print('Running SLC20 experiment...')
    run_slc20_experiment(args)
    print('SLC20 experiment finished.')

def simple_test_suite(args):
    '''
    Runs through a short test suite, running each test or not depending
    on the arguments specified in the command line (or through defaults!)

    I don't think pytest / similar is worth it, so instead trying this.
    '''

    testing_run = False

    if args.test_compute_fvec:
        fvec = util.compute_fvec('../data/test.txt')
        print(fvec)
        assert fvec[:6] == [28.5, 13, 0, 2, -1.5, 1]
        testing_run = True
    if args.test_compute_fvecs_for_parts:
        parts = ['LA13272', 'MV01950']
        X = util.compute_fvecs_for_parts(parts)
        print(X)
        testing_run = True
    if args.test_load_scores:
        scores_dict = util.load_participant_scores('../data/participant_data.csv')
        print(scores_dict)
        testing_run = True

    if testing_run:
        print('Tests completed; exiting without running experiments...')
        exit()

# expose CLI
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_compute_fvec', default=False,
            help='Flag indicating whether to test computation of the feture vectors')
    parser.add_argument('--test_compute_fvecs_for_parts', default=False,
            help='Flag indicating whether to test computation of specified features')
    parser.add_argument('--test_load_scores', default=False,
            help='Flag indicating whether to test loading of response variables')
    parser.add_argument('--random-seed', default=1, type=int,
            help='Random seed to ensure replicable results.')
    args = parser.parse_args()

    # Set random seeds so results are replicable
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Optionally run some sanity-check tests as specified via CLI args.
    # Experiments are not run if tests are run.
    simple_test_suite(args)

    # The main event -- run some experiment.
    main(args)

    
