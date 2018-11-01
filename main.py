

# JOHN
def fitGAD7():
    """ Fits a regression model to predict GAD7 anxiety scores based on 
    N participants' head movement data.
    
    Returns a tuple (X, Y, Theta) where X is a N-by-120 numpy array,
    Y is a N-by-1, and Theta is 120-by-1.

    Note: we can change this architecture up - I'm open to suggestions :)
    """
    pass

# SARAH / COOPER
def fitSCL20():
    """ Fits a classification model to predict major depressive disorder
    based on N participants' head movement data.
    
    Returns a tuple (X, Y, Theta) where X is a N-by-120 numpy array,
    Y is a N-by-1, and Theta is 120-by-1.

    Note: we can change this architecture up - I'm open to suggestions :)
    """
    pass
    
# JOHN / all
def main():
    """ Fits models to predict mental health outcomes (anxiety, 
    depression) based on head movement data gathered during various
    virtual reality experiences.

    Analyzes these models' efficacy on a test set and (maybe) on 
    future (2 month, 6 month, 12 month) data.
    """
    pass
   

# expose CLI
if __name__ == '__main__':
    main()
    
