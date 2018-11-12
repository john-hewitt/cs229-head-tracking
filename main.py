from argparse import ArgumentParser
import random

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import util
import models
import experiments

def main(args):
    """ 
    
    Fits models to predict mental health outcomes (anxiety, 
    depression) based on head movement data gathered during various
    virtual reality experiences.

    Analyzes these models' efficacy on a test set and (maybe) on 
    future (2 month, 6 month, 12 month) data.
    """

    experiment_name = args.expt
    experiments.run_experiment(experiment_name, args)

    exit()
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
    if args.test_which_tracking:
        pid_mos_t = util.which_parts_have_tracking_data('../data/Tracking/')
        print('first 5 items: {}'.format(pid_mos_t[:5]))
        print('number of items: {}'.format(len(pid_mos_t)))
        testing_run = True
    if args.test_which_scores:
        pid_mos_sg = util.which_parts_have_score('../data/participant_data.csv', 'GAD7_score')
        pid_mos_ss = util.which_parts_have_score('../data/participant_data.csv', 'SCL_20')
        print('first 5 items: {}'.format(pid_mos_sg[:5]))
        print('number of items: {}'.format(len(pid_mos_sg)))
        testing_run = True
    if args.test_which_scores and args.test_which_tracking:
        print('number of (pid, mo) pairs that have both tracking data and GAD7 scores: {}'.format(len(set(pid_mos_t) & set(pid_mos_sg))))
        print('number of (pid, mo) pairs that have both tracking data and SCL20 scores: {}'.format(len(set(pid_mos_t) & set(pid_mos_ss))))
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
    parser.add_argument('--test_which_tracking', default=False,
            help='flag indicating whether to test loading of which (pid, mo) pairs have tracking data')
    parser.add_argument('--test_which_scores', default=False,
            help='flag indicating whether to test loading which (pid, mo) pairs have GAD7/SCL20 scores')
    parser.add_argument('--random-seed', default=1, type=int,
            help='Random seed to ensure replicable results.')
    parser.add_argument('--expt', type=str, default = '',
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

    
