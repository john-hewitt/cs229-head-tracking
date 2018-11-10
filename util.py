import csv
import json
import os
import numpy as np
import sklearn as sk

# globals
mos = [0, 2, 6, 12] 
exps = ['R', 'N1', 'N2', 'P1', 'P2']

# file naming conventions
def tfname_to_id(fname):
    return fname[9:15]
def tfname_to_mo(fname):
    pass
def tfname_to_exp(fname):
    pass

def tracking_file(part, mo, exp):
    """ Helper function.
    
        For a given participant PART, time frame (month number) MO, 
        and experience type EXP, returns the string associated with 
        the relevant .txt filename
    """
    valid_pId = lambda x : len(x) == 7 # good enough

    assert valid_pId(part)
    assert mo in mos 
    assert exp in exps

    if mo > 0:
        tfilename = 'tracking_{}{}{}.txt'.format(part, int(mo), exp).upper()
    else:
        tfilename = 'tracking_{}{}.txt'.format(part, exp).upper()

    base = os.path.relpath('../data/Tracking/')
    tfile = os.path.join(base, tfilename)

    return tfile

def which_months(part):
    """ Helper function

        Returns all of the months for which we have the head tracking
        data of participant PART for all experience types.
    """
    return filter(lambda mo: have_part_mo(part, mo), mos)

def have_part_mo(part, mo):
    """ Helper function 
        
        Returns True if we have month MO's head tracking 
        data of participant PART for all experience types.

        Returns False otherwise.
    """
    tfiles = [tracking_file(part, mo, exp) for exp in exps]
    return all([os.path.isfile(tfile) for tfile in tfiles]) 

# SARAH
def load_participant_scores(csvfile):
    """ Load participant data (GAD7 and SCL20 scores) from CSVFILE.

        Only load a participant's data if we have their head tracking
        data. Useful helper function: have_part_mo.

        Returns a dictionary mapping participant ID string to 
        a tuple (GAD7 score, SCL20 score).
    """

    scores_dict = {}

    # load labels from file                                                
    with open(csvfile,'rt', encoding = "utf8") as csvfile:

        reader = csv.DictReader(csvfile)   
        next(reader) # skip headings

        for row in reader:
            part = row["subNum"]
            for mo in which_months(part):
                gad7 = row["GAD7_score"]
                scl20 = row["SCL_20"]
                # only add labels if both scores are valid
                if (gad7 != "NA") and (scl20 != "NA" ):
                    scores_dict[part] = (gad7, scl20)

    return scores_dict

gad7 = 'GAD7_score'
scl20 = 'SCL_20'
def load_scores(csvfile, part_mos, score_type):
    """ Given a list of tuples PART_MOS (the first element is participant
        id, the second is the month), load the SCORE_TYPE score for
        each from CSVFILE.

        SCORE_TYPE is the name of the column that contains the score.

        Returns a len(part_mos)-by-1 numpy array.
    """
    pass

def which_parts_have_score(csvfile, scoreType):
    """ For scoring metric SCORETYPE, return all tuples (participant id, 
        month) for which we have that score in CSVFILE.

        SCORE_TYPE is the name of the column that contains the score.

        Returns a list of tuples.
    """
    pass


def which_parts_have_tracking_data(folder):
    """ Returns all tuples (participant id, month) for which we have 
        tracking data in FOLDER. 
    """
    pass


# SARAH
def GAD7_labels(parts, scoresDict):
    """ For each of N participants given by PARTS, determine the GAD7 
        score label.

        Returns the labels as a N-by-1 numpy array.
    """

    gad_labels = np.zeros(len(parts))

    for part, part_idx in zip(parts, range(0, len(parts))):        
        gad_labels[part_idx] = scoresDict[part][0]

    return gad_labels
    
# SARAH
def SCL20_labels(parts, scoresDict):
    """ For each of N participants given by PARTS, determine the SCL20
        score label.

        Returns the labels as a N-by-1 numpy array.
    """

    scl_labels = np.zeros(len(parts)) 
    
    for part, part_idx in zip(parts, range(0, len(parts))):
        scl_labels[part_idx] = scoresDict[part][1]
        
    # return actual scl score or {0,1} label?
    # if want y in {0,1}:
    scl_labels = SCL20_threshold(scl_labels)

    return scl_labels

    
def SCL20_threshold(scores):
    scores[scores < 0.5] = 0
    scores[scores >= 0.5] = 1
    return scores


def compute_fvec(tfile):
    """ Takes in a tracking file path, and computes the feature vector
        corresponding to head movement data for each experience type. 
        The features are as described in meeting.
        
        There are 12 features / channel * 2 channels = 24 features.

        Returns the feature vector as a 1-by-24 numpy array.
    """
    # load data from file
    with open(tfile) as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')
        
        rot = [] 
        for row in tsvin:
            ch1 = row[1]
            ch2 = row[2]
            roti = json.loads(ch1) + json.loads(ch2)
            rot.append(roti)

    # compute features
    rot = np.array(rot)
    rotmus = np.mean(rot, axis=0)
    rotsigmas = np.var(rot, axis=0) 

    delta = np.absolute(np.diff(rot, axis=0))
    deltasums = np.sum(delta, axis=0)
    deltasigmas = np.var(delta, axis=0)

    # shape output
    fvec = np.concatenate([rotmus, rotsigmas, deltasums, deltasigmas])
    fvec = np.expand_dims(fvec, 0)
    return fvec


def compute_fvecs_for_parts(parts, baseline_only=False):
    """ For each of participants given by PARTS, compute features for
        each of the experience types and concatenate them to form
        one feature vector per participant.

        If BASELINE_ONLY flag is specified, compute the features from
        only the baseline data of the participants, asserting that 
        we have the full set of baseline data for each. In this case,
        the N described below would be len(PARTS).

        There are 24 features / experience * 5 experiences = 120 features

        Returns the training matrix as an N-by-120 numpy array, where
        N is the number of full sets of VR data we have on the given
        PARTS.
    """
    fvecs = None
    for part in parts:
        if baseline_only:
            baseline_mo = 0
            assert have_part_mo(part, baseline_mo)
            tfiles = [tracking_file(part, baseline_mo, exp) for exp in exps]
        else:
            part_mos = which_months(part)
            tfiles = [tracking_file(part, mo, exp) for exp in exps
                                                   for mo in part_mos]
        expvecs = [compute_fvec(tfile) for tfile in tfiles] 
        fvec = np.concatenate(expvecs, axis=1)
        if fvecs is None: 
            fvecs = np.array(fvec)
        else:
            fvecs = np.concatenate([fvecs, fvec], axis=0)
        
    return fvecs
    

    
    
    


