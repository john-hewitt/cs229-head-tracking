import csv
import numpy as np

# globals
mos = [0, 2, 6, 12] 
exps = ['R', 'N1', 'N2', 'P1', 'P2']

def tracking_file(pId, mo, exp):
    """ Helper function.
    
        For a given participant id PID, time frame (month number) MO, 
        and experience type EXP, returns the string associated with 
        the relevant .txt filename
    """
    valid_pId = lambda x : len(x) == 7 # good enough

    assert valid_pId(pId)
    assert mo in mos 
    assert exp in exps

    tfilename = 'tracking_{}{}{}.txt'.format(pId, int(mo), exp)

    base = os.relpath('../data/Tracking/')
    tfile = os.path.join(base, tfilename)

    return tfile

# SARAH
def load_participant_scores(csvfile):
    """ Load participant data (GAD7 and SCL20 scores) from CSVFILE.

        Returns a dictionary mapping participant ID string to 
        a tuple (GAD7 score, SCL20 score).
    """
    pass

# SARAH
def GAD7_labels(parts, scoresDict):
    """ For each of N participants given by PARTS, determine the GAD7 
        score label.

        Returns the labels as a N-by-1 numpy array.
    """
    pass
    
# SARAH
def SCL20_labels(parts, scoresDict):
    """ For each of N participants given by PARTS, determine the SCL20
        score label.

        Returns the labels as a N-by-1 numpy array.
    """
    pass

# COOPER
def compute_fvec(tfile):
    """ Takes in a tracking file path, and computes the feature vector
        corresponding to head movement data for each experience type. 
        The features are as described in meeting.
        
        There are 12 features / channel * 2 channels = 24 features.

        Returns the feature vector as a 1-by-24 numpy array.
    """
    pass

# COOPER
def compute_fvecs_for_parts(parts):
    """ For each of N participants given by PARTS, compute features for
        each of the experience types and concatenate them to form
        one feature vector per participant.

        There are 24 features / experience * 5 experiences = 120 features

        Returns the training matrix as an N-by-120 numpy array.
    """
    pass

    
    
    


