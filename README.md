# cs229-head-tracking
Code for analyzing mental health outcomes with VR head tracking data

## Getting Started

In the python virtual environment of your choice, run the following
to install the necessary python packages for this dataset:

        pip install -r requirements.txt

To run a few sanity check tests, run `main.py` with test flags as follows:
(if any test flag is set, no experiments will be run after the tests complete.)

        python main.py --test_compute_fvec 1 --test_compute_fvecs_for_parts 1 --test_load_scores 1 

To run an experiment without the testing, omit the flags above, running

        python main.py --expt EXPERIMENT_NAME

where `EXPERIMENT_NAME` is literally the name of a function in `experiments.py` where you've written
the code necessary to run a whole experiment.

### Experimental design

We've split the data into a training set and a test set (by participant index.) We have 34 training examples in the first month and 8 testing examples.
The experiment code template I've set in `experiments.py` uses hold-one-out evaluation on the training set.
We will not look at the test until we're done with development.

## Code Layout

 - `main.py`: command-line interface; calls other modules to run experiments.
 - `util.py`: Useful helper functions and data loading/saving
 - `models.py`: Self-sufficient ML classes that permit fitting/predicting on
 datasets.
 - `experiments.py`: Experiment functions: each one holds the logic for running an experiment of interest, logging its results.
    Currently contains 1 model for GAD7 and 1 model for SLC20.
 - `constants.py`: contains certain global constants, like how we're splitting the participants into train/test sets.

## Experiment Parameters

For a quick overview of the paramters variable via the command line in this experiment, run

        python main.py --help

See below for a more detailed description of each parameter.

- `--random-seed`: This specifies an integer random seed to start the pseudorandom process
of both `numpy` and `python`. (Note that `sklearn` simply uses numpy's psuedorandom process,
so no need to set that.) This is set to 1 by default; re-running with the same seed should
lead to identical results.
- `--test*`: All flag of this form refer to code testing procedures. Refer to their `--help`
strings for more information. No tests are run by default.

- `--featurization`: This specifies which type of featurization to perform.
 The valid options are `summary_stats` and `norm_hist`. Summary stats provides the variance and length-normalized sum of head radians traveled for each of roll/pitch/yaw for each of 2 sensors. The `norm_hist`, or normalized histogram, provides something of a delta-histogram. First the absolute value of the delta between each row of head positions is taken. Then this is bucketed (somewhat arbitrarily) into discrete buckets. I chose the buckets to have reasonable variation -- it's a log scale. then, for each of roll/pitch/yaw for each of the 2 sensors, the number of deltas in each bucket is computed. These values are normalized by the total number of timesteps. Viola . The idea here is that if a large percent of your head movements are high-velocity (large delta) then you are possibly anxious.
