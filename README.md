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

        python main.py

## Code Layout

 - `main.py`: command-line interface; calls other modules to run experiments.
 - `util.py`: Useful helper functions and data loading/saving
 - `models.py`: Self-sufficient ML classes that permit fitting/predicting on
 datasets.
    Currently contains 1 model for GAD7 and 1 model for SLC20.

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
