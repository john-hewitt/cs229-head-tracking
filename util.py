import csv
import json
import os
import numpy as np
import sklearn as sk
import re

# globals
mos = [0, 2, 6, 12] 
exps = ['R', 'N1', 'N2', 'P1', 'P2']

# file naming conventions
id_reg = '[a-z]{2}[0-9]{5}'
mo_reg = '(((2)|(6)|(12))mo)?'
exp_reg = '((n1)|(n2)|(r)|(p1)|(p2))'
tfname_reg = r'tracking_{}{}{}\.txt'.format(id_reg,
                                           mo_reg,
                                           exp_reg)
def valid_tfname(fname):
    fname = fname.lower()
    val = re.match(tfname_reg, fname) is not None
    return val

def tfname_parts(fname):
    assert valid_tfname(fname)
    fname = fname.lower()
    Id = fname[9:16]
    if fname[17:19] == 'mo':
        Mo = int(fname[16])
    elif fname[18:20] == 'mo':
        Mo = 12
    else:
        Mo = 0
    if fname[-5] == 'r':
        Exp = 'r'
    else:
        Exp = fname[-6:-4]
    assert Exp.upper() in exps
    assert Mo in mos
    return Id,Mo,Exp

def tracking_file(part, mo, exp):
    """ Helper function.
    
        For a given participant PART, time frame (month number) MO, 
        and experience type EXP, returns the string associated with 
        the relevant .txt filename
    """
    valid_pId = lambda x : len(x) == 7 # good enough

    assert valid_pId(part)
    assert mo in mos 
    assert exp.upper() in exps

    if mo > 0:
        tfilename = 'tracking_{}{}mo{}.txt'.format(part, mo, exp).upper()
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
    have = all([os.path.isfile(tfile) for tfile in tfiles]) 
    return have

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
            if have_part_mo(part, 0):
                gad7 = row["GAD7_score"]
                scl20 = row["SCL_20"]
                # only add labels if both scores are valid
                if (gad7 != "NA") and (scl20 != "NA" ):
                    scores_dict[part] = (gad7, scl20)

    return scores_dict

gad7 = 'GAD7_score'
scl20 = 'SCL_20'
def load_scores(csvfile, pid_mos, score_type):
    """ Given a list of tuples PID_MOS (the first element is participant
        id, the second is the month), load the SCORE_TYPE score for
        each from CSVFILE.

        SCORE_TYPE is the name of the column that contains the score.

        Note: this function should only be passed pid_mo pairs for
        which we have the given score for (use which_parts_have_score).

        Returns a len(part_mos)-by-1 numpy array.
    """
    scores_dict = {}
    with open(csvfile,'rt', encoding = "utf8") as csvfile:

        reader = csv.DictReader(csvfile)   
        next(reader) # skip headings

        for row in reader:
            pid = row["subNum"].lower()
            mo = int(row["time"])
            if (pid,mo) in pid_mos:
                score = row[score_type]
                assert score != "NA"
                scores_dict[(pid,mo)] = int(score)

    scores = [scores_dict[pid_mo] for pid_mo in pid_mos]
    return scores

def which_parts_have_score(csvfile, score_type):
    """ For scoring metric SCORETYPE, return all tuples 
        (lowercase participant id, month) for which we have that score 
        in CSVFILE.

        SCORE_TYPE is the name of the column that contains the score.

        Returns a list of tuples.
    """
    pid_mos = [] 
    with open(csvfile,'rt', encoding = "utf8") as csvfile:
        reader = csv.DictReader(csvfile)   
        next(reader) # skip headings

        for row in reader:
            pid = row['subNum'].lower()
            mo = int(row['time'])
            score = row[score_type]
            if score != "NA":
                pid_mos.append((pid, mo))

    # ensure each element of pid_mos is unique
    assert len(set(pid_mos)) == len(pid_mos)

    return pid_mos

def which_parts_have_tracking_data(folder, verbose=False):
    """ Returns all tuples (lowercase participant id, month) for which 
        we have tracking data in FOLDER. 

    """
    vprint = print if verbose else lambda x : x

    # get lowercase name of all fles
    tfiles = [f.lower() for f in os.listdir(folder)]
    vprint('number of tracking files found: {}'.format(len(tfiles)))

    # regex-filter for all "usable" filenames
    val_tfiles = filter(valid_tfname, tfiles)

    # parse filenames to generate tuples
    pid_mos = [tfname_parts(f)[0:2] for f in val_tfiles]
    vprint('number of VALID tracking files found: {}'.format(len(pid_mos)))
    pid_mos_uniq = list(set(pid_mos))
    vprint('number of (pid,mo) pairs found: {}'.format(len(pid_mos_uniq)))
    # make sure all returned pairs have all experience types
    pid_mos_filt = list(filter(lambda pm : have_part_mo(*pm), pid_mos_uniq))

    return pid_mos_filt


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


def compute_fvecs_for_parts(pid_mos):
    """ For each (pid, month) given by PID_MOS, compute features for
        each of the experience types and concatenate them to form
        one feature vector per participant. 
        
        Note: We must have tracking data across all 5 experience types 
        for each (pid, month) pair.

        There are 24 features / experience * 5 experiences = 120 features

        Returns the training matrix as an N-by-120 numpy array, where
        N is the number of full sets of VR data we have on the given
        PARTS.
    """
    fvecs = None
    for pid, mo in pid_mos:
        tfiles = [tracking_file(pid, mo, exp) for exp in exps]

        expvecs = [compute_fvec(tfile) for tfile in tfiles] 
        fvec = np.concatenate(expvecs, axis=1)
        if fvecs is None: 
            fvecs = np.array(fvec)
        else:
            fvecs = np.concatenate([fvecs, fvec], axis=0)
        
    return fvecs
    

def get_experience_indices(experience):
    """Given an experience type in ['R', 'N1', 'N2', 'P1', 'P2']  
       return the indices in the feature vector that the 
       experience maps to 
    """
    if experience == 'R':
        indices = (0, 23)
    elif experience == 'N1':
        indices = (24, 47)
    elif experience == 'N2':
        indices = (48, 71)
    elif experience == 'P1':
        indices =  (72, 95)
    elif experience ==  'P2':
        indices = (96, 123)
    return indices
    
    


