import util
from sklearn import linear_model

# JOHN
def fitGAD7():
    """ Fits a regression model to predict GAD7 anxiety scores based on 
    N participants' head movement data.
    
    Returns a tuple (X, Y, Theta) where X is a N-by-120 numpy array,
    Y is a N-by-1, and Theta is 120-by-1.

    Note: we can change this architecture up - I'm open to suggestions :)
    """
    tracking_data = '../data/test.txt'
    part_data = '../data/participant_data.csv'
    
    # get PARTS 
    # parts = 
    # for now..
    parts = ['LA13272', 'LA14016', 'MV00962', 'MV01113', 'MV01950', 'MV07296', 'MV07303','MV07647','MV08032','MV09122', 'MV09305', 'MV09441', 'MV09586','MV11133','MV11150', 'MV11202', 'PA22014', 'PA22544','PA22561','PA22728','PA23284', 'PA23955','PA24326', 'PA24859','PA24876','PA25084','PA25119',  'PA25306','PA26203','PA26376', 'PA26623' 'PA27784','PA27793','PA27962','PA30895', 'PA30677', 'PA30862', 'PA30895', 'SU30734', 'SU30816','SU33550','SU35282']

    # load features
    train_matrix = util.compute_fvecs_for_parts(parts)

    # load labels
    score_dict = util.load_participant_scores(part_data)
    
    # get gad7 labels
    gad_labels = util.GAD7_labels(score_dict)

    # train linear regression model with lasso regularization
    clf = linear_model.Lasso(alpha = 0.1)
    clf.fit(train_matrix, gad_labels)

    # get theta
    theta_coeff = clf.coef_
    return (X, y, theta_coeff)



# SARAH / COOPER
def fitSCL20():
    """ Fits a classification model to predict major depressive disorder
    based on N participants' head movement data.
    
    Returns a tuple (X, Y, Theta) where X is a N-by-120 numpy array,
    Y is a N-by-1, and Theta is 120-by-1.

    Note: we can change this architecture up - I'm open to suggestions :)
    """
    tracking_data = '../data/test.txt'
    part_data = '../data/participant_data.csv'

    # get PARTS                                                                                           
    # parts =                                                                                                                                                                                               
    # load features                                                                          
    train_matrix = util.compute_fvecs_for_parts(parts)

    # load labels                                                                                     
    score_dict = util.load_participant_scores(part_data)

    # get scl20 labels  - 0 or 1                                                        
    scl_labels = util.SCL20_labels(parts, score_dict)

    
    pass
    
# JOHN / all
def main():
    """ Fits models to predict mental health outcomes (anxiety, 
    depression) based on head movement data gathered during various
    virtual reality experiences.

    Analyzes these models' efficacy on a test set and (maybe) on 
    future (2 month, 6 month, 12 month) data.
    """ 
                          
    fitGAD7()

    #pass
   

# expose CLI
if __name__ == '__main__':
    test_compute_fvec = False
    test_compute_fvecs_for_parts = True
    test_load_scores = True
    if test_compute_fvec:
        fvec = util.compute_fvec('../data/test.txt')
        print(fvec)
        assert fvec[:6] == [28.5, 13, 0, 2, -1.5, 1]
    if test_compute_fvecs_for_parts:
        parts = ['LA13272', 'MV01950']
        X = util.compute_fvecs_for_parts(parts)
        print(X)
    if test_load_scores:
        scores_dict = util.load_participant_scores('../data/participant_data.csv')
        print(scores_dict)
    main()

    
