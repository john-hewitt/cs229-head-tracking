import random

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import itertools

import util
import json
import models
import experiments
from constants import test_participants
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import mode
import seaborn as sns
import json
sns.set(style="darkgrid")


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

class GAD7AutoHyperparameterOptimization():
  '''
  Class for conducting automatic hyperparameter optimization
  where each hyperparameter is an ``expert" trained off of different data.
  '''
  def __init__(self, args, models=None):
    self.args = args
    self.model_count = len(list(self.model_retriever()))
    self.best_weights = [1 for x in models]
    self.trial_set = []
    self.trials_per_study = 10

  def optimize(self, studies=200):
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

    X_train_dev_sstat = util.compute_fvecs_for_parts(pid_mos_use, 'summary_stats')#[:,1:10]
    X_train_dev_dft = util.compute_fvecs_for_parts(pid_mos_use, 'dft')#[:,1:10]
    scores = util.load_scores(part_data, pid_mos_use, util.gad7)
    y_train_dev = np.array(scores) >= 10

    #X_train_dev_sstat = sklearn.preprocessing.normalize(X_train_dev_sstat, axis=1)
    #X_train_dev_dft = sklearn.preprocessing.normalize(X_train_dev_dft, axis=1)

    for study_index in range(studies):
      trials = []
      weights, threshold = self.sample_hyperparameters()
      for rep in range(5):
        print('Trying weights {}, threshold {}'.format(weights, threshold))
        results = gad7_final_dev(self.args, 
            models.LearnedEnsembleClassificationGAD7Model(self.model_retriever, weights, threshold), X_train_dev_sstat, X_train_dev_dft, y_train_dev)
        #results = gad7_kfold_classification(self.args, 
        #    prespecified_model=models.LearnedEnsembleClassificationGAD7Model(self.model_retriever, weights, threshold))
        #self.trial_set.append((weights, threshold, results['model_f1'], results['model_precision'], results['model_recall']))
        trials.append((results['model_f1'], results['model_precision'], results['model_recall']))
      f1 = np.mean([x[0] for x in trials])
      precision = np.mean([x[1] for x in trials])
      recall = np.mean([x[2] for x in trials])
      print('AVERAGE f1, prec, recall', f1, precision, recall)
      with open('trials3.jsonl', 'a') as fout:
        weights = list(weights)
        json.dump((weights, threshold, {'model_f1':f1, 'model_precision': precision, 'model_recall':recall}), fout)
        fout.write('\n')
        #print(results)
      #input()

  def sample_hyperparameters(self):
    #weights = np.random.normal(size=self.model_count)
    weights = np.random.gamma(2, scale=1.0, size=self.model_count*2)
    weight_sum = np.sum(weights)
    weights = weights / weight_sum
    #threshold = 1
    #threshold = np.random.uniforembed m(low=.3, high=.7)
    threshold = np.random.uniform(low=.4, high=.6)
    return weights, threshold

  @staticmethod
  def model_retriever():
    yield LogisticRegression(max_iter=1000000, solver='liblinear', C=1)
    yield MultinomialNB()
    yield DecisionTreeClassifier(max_depth=5)


def gad7_kfold_classification_hpo(args):
  hpo_model = GAD7AutoHyperparameterOptimization(args, [])
  hpo_model.optimize()

def freq_analysis(args):
    tfile = '../data/Tracking/tracking_30677R.txt'
    tfile = '../data/Tracking/tracking_MV0087812MOR.txt'

    fvec = util.compute_freq_fvec(tfile) 
    print(fvec)

    fs = 1000
    N = 20
    dfs = np.fft.fftfreq(N, d=1/float(fs))

    data, diffs = util.load_channels(tfile)
    data_f = np.fft.fft(data, n=N, axis=0)
    diffs_f = np.fft.fft(diffs, n=N, axis=0)
    plt.subplot(2,2,1)
    plt.title('Data')
    for i in range(3):
        plt.plot(data[:,i], alpha=.8)
    plt.subplot(2,2,2)
    plt.title('DFT of data')
    for i in range(3):
        plt.plot(dfs, np.absolute(data_f[:,i]), alpha=.8)
    plt.legend(['yaw','pitch','roll'])
    plt.subplot(2,2,3)
    plt.title('Deltas')
    for i in range(3):
        plt.plot(diffs[:,i], alpha=.8)
    plt.subplot(2,2,4)
    plt.title('DFT of deltas')
    for i in range(3):
        plt.plot(dfs, np.absolute(diffs_f[:,i]), alpha=.8)
    plt.show()


def gad7_final_test(args):
  weights = []
  threshold = []

  tracking_data = '../data/Tracking/'
  part_data = '../data/participant_data.csv'

  # load usable (pid, mo) pairs, and make sure to remove test set
  pid_mos_sg = util.which_parts_have_score(part_data, util.gad7)
  pid_mos_t = util.which_parts_have_tracking_data(tracking_data)
  pid_mos_both = list(set(pid_mos_sg) & set(pid_mos_t))
  pid_mos_use = list(filter(lambda pm : pm[0].upper() not in test_participants, pid_mos_both))
  pid_mos_test =list(filter(lambda pm : pm[0].upper() in test_participants, pid_mos_both)) 
  #pid_mos_use = list(filter(lambda pm: pm[1] == 0, pid_mos_use))
  print('Loaded {} (pid, mo) pairs with both tracking data and GAD7 scores.'.format(len(pid_mos_both)))
  print('Removed {} (pid, mo) test set pairs to leave {} total to train with.'.format(len(pid_mos_both) - len(pid_mos_use), len(pid_mos_use)))
  print('Loaded {} (pid, mo) test set pairs to leave {} total to test on.'.format(len(pid_mos_both) - len(pid_mos_use), len(pid_mos_use)))


  X_train_dev_sstat = util.compute_fvecs_for_parts(pid_mos_use, 'summary_stats')#[:,1:10]
  X_train_dev_dft = util.compute_fvecs_for_parts(pid_mos_use, 'dft')#[:,1:10]
  scores = util.load_scores(part_data, pid_mos_use, util.gad7)
  y_train_dev = np.array(scores) >= 10

  test_scores = util.load_scores(part_data, pid_mos_test, util.gad7)
  X_test_sstat = util.compute_fvecs_for_parts(pid_mos_test, 'summary_stats')#[:,1:10]
  X_test_dft = util.compute_fvecs_for_parts(pid_mos_test, 'dft')#[:,1:10]
  y_test = np.array(test_scores) >= 10

  #### Each individual learner
  f1_results = []
  precision_results = []
  recall_results = []
  for i in range(50):
    prediction_model = LogisticRegression(max_iter=1000000, solver='liblinear', C=1)
    prediction_model.fit(X_train_dev_sstat, y_train_dev)
    predictions = prediction_model.predict(X_test_sstat)
    f1_results.append(sklearn.metrics.f1_score(y_test, predictions))
    precision_results.append(sklearn.metrics.precision_score(y_test, predictions))
    recall_results.append(sklearn.metrics.recall_score(y_test, predictions))
  print('LogReg sstat achieved {} f1, {} precision, {} recall'.format(np.mean(f1_results), np.mean(precision_results), np.mean(recall_results)))
  print()
    
  f1_results = []
  precision_results = []
  recall_results = []
  for i in range(50):
    prediction_model = LogisticRegression(max_iter=1000000, solver='liblinear', C=1)
    prediction_model.fit(X_train_dev_dft, y_train_dev)
    predictions = prediction_model.predict(X_test_dft)
    f1_results.append(sklearn.metrics.f1_score(y_test, predictions))
    precision_results.append(sklearn.metrics.precision_score(y_test, predictions))
    recall_results.append(sklearn.metrics.recall_score(y_test, predictions))
  print('LogReg dft achieved {} f1, {} precision, {} recall'.format(np.mean(f1_results), np.mean(precision_results), np.mean(recall_results)))
  print()

  f1_results = []
  precision_results = []
  recall_results = []
  for i in range(50):
    prediction_model = MultinomialNB()
    prediction_model.fit(X_train_dev_sstat, y_train_dev)
    predictions = prediction_model.predict(X_test_sstat)
    f1_results.append(sklearn.metrics.f1_score(y_test, predictions))
    precision_results.append(sklearn.metrics.precision_score(y_test, predictions))
    recall_results.append(sklearn.metrics.recall_score(y_test, predictions))
  print('MultinomialNB sstat achieved {} f1, {} precision, {} recall'.format(np.mean(f1_results), np.mean(precision_results), np.mean(recall_results)))
  print()
    
  f1_results = []
  precision_results = []
  recall_results = []
  for i in range(50):
    prediction_model = MultinomialNB()
    prediction_model.fit(X_train_dev_dft, y_train_dev)
    predictions = prediction_model.predict(X_test_dft)
    f1_results.append(sklearn.metrics.f1_score(y_test, predictions))
    precision_results.append(sklearn.metrics.precision_score(y_test, predictions))
    recall_results.append(sklearn.metrics.recall_score(y_test, predictions))
  print('MultinomialNB dft achieved {} f1, {} precision, {} recall'.format(np.mean(f1_results), np.mean(precision_results), np.mean(recall_results)))
  print()

  f1_results = []
  precision_results = []
  recall_results = []
  for i in range(50):
    prediction_model = DecisionTreeClassifier(max_depth=5)
    prediction_model.fit(X_train_dev_sstat, y_train_dev)
    predictions = prediction_model.predict(X_test_sstat)
    f1_results.append(sklearn.metrics.f1_score(y_test, predictions))
    precision_results.append(sklearn.metrics.precision_score(y_test, predictions))
    recall_results.append(sklearn.metrics.recall_score(y_test, predictions))
  print('DecisionTree sstat achieved {} f1, {} precision, {} recall'.format(np.mean(f1_results), np.mean(precision_results), np.mean(recall_results)))
  print()
    
  f1_results = []
  precision_results = []
  recall_results = []
  for i in range(50):
    prediction_model = DecisionTreeClassifier(max_depth=5)
    prediction_model.fit(X_train_dev_dft, y_train_dev)
    predictions = prediction_model.predict(X_test_dft)
    f1_results.append(sklearn.metrics.f1_score(y_test, predictions))
    precision_results.append(sklearn.metrics.precision_score(y_test, predictions))
    recall_results.append(sklearn.metrics.recall_score(y_test, predictions))
  print('DecisionTree dft achieved {} f1, {} precision, {} recall'.format(np.mean(f1_results), np.mean(precision_results), np.mean(recall_results)))
  print()


  model_retriever = GAD7AutoHyperparameterOptimization.model_retriever

  #### Precision-weighted ensemble
  #precision_weights = (0.09386384861806127, 0.18993921165508632, 0.2392826714504859, 0.10884956328744151, 0.18285384299578894, 0.1852108619931362)
  precision_weights = (0.2520549926904262, 0.07816005770204427, 0.15988226432360003, 0.14505573023656232, 0.15213310641966477, 0.21271384862770248)
  precision_threshold = 0.576689940593974
  f1_results = []
  precision_results = []
  recall_results = []
  for i in range(50):
    prediction_model = models.LearnedEnsembleClassificationGAD7Model(model_retriever, precision_weights, precision_threshold)
    prediction_model.fit(X_train_dev_sstat, X_train_dev_dft, y_train_dev)
    predictions = prediction_model.predict(X_test_sstat, X_test_dft)
    f1_results.append(sklearn.metrics.f1_score(y_test, predictions))
    precision_results.append(sklearn.metrics.precision_score(y_test, predictions))
    recall_results.append(sklearn.metrics.recall_score(y_test, predictions))
  print('Precision-Weighted Ensemble achieved {} f1, {} precision, {} recall'.format(np.mean(f1_results), np.mean(precision_results), np.mean(recall_results)))
  print()


  #### Recall-weighted ensemble
  #recall_weights = (0.245132576228378, 0.1861073615849133, 0.17776944884823678, 0.06892088169721604, 0.14523392272502403, 0.17683580891623188)
  #recall_threshold = 0.31918712321972736
  recall_weights = (0.03991128026289153, 0.504535178322661, 0.06468635218243506, 0.167540125009916, 0.13672334732710784, 0.08660371689498858)
  recall_threshold = 0.49644076609820564
  f1_results = []
  precision_results = []
  recall_results = []
  for i in range(50):
    prediction_model = models.LearnedEnsembleClassificationGAD7Model(model_retriever, recall_weights, recall_threshold)
    prediction_model.fit(X_train_dev_sstat, X_train_dev_dft, y_train_dev)
    predictions = prediction_model.predict(X_test_sstat, X_test_dft)
    f1_results.append(sklearn.metrics.f1_score(y_test, predictions))
    precision_results.append(sklearn.metrics.precision_score(y_test, predictions))
    recall_results.append(sklearn.metrics.recall_score(y_test, predictions))
  print('Recall-Weighted Ensemble achieved {} f1, {} precision, {} recall'.format(np.mean(f1_results), np.mean(precision_results), np.mean(recall_results)))
  print()

  #### F1-weighted ensemble
  #f1_weights = (0.11340094317606973, 0.12344507465738329, 0.3138537938592019, 0.19013927384687196, 0.13686757174378053, 0.12229334271669257)
  #f1_threshold = 0.4200003067247799
  f1_weights = (0.2929445494249835, 0.339263074997816, 0.11693681879509694, 0.09380796340733755, 0.02920737103378879, 0.12784022234097714)
  f1_threshold = 0.44696411339368014
  f1_results = []
  precision_results = []
  recall_results = []
  for i in range(50):
    prediction_model = models.LearnedEnsembleClassificationGAD7Model(model_retriever, f1_weights, f1_threshold)
    prediction_model.fit(X_train_dev_sstat, X_train_dev_dft, y_train_dev)
    predictions = prediction_model.predict(X_test_sstat, X_test_dft)
    f1_results.append(sklearn.metrics.f1_score(y_test, predictions))
    precision_results.append(sklearn.metrics.precision_score(y_test, predictions))
    recall_results.append(sklearn.metrics.recall_score(y_test, predictions))
  print('F1-Weighted Ensemble achieved {} f1, {} precision, {} recall'.format(np.mean(f1_results), np.mean(precision_results), np.mean(recall_results)))
  print()

  #### Equal-weighted-ensemble
  equal_weights = (.166666,.166666,.166666,.166666,.166666,.166666)
  half_threshold = .5
  f1_results = []
  precision_results = []
  recall_results = []
  for i in range(50):
    prediction_model = models.LearnedEnsembleClassificationGAD7Model(model_retriever, equal_weights, half_threshold)
    prediction_model.fit(X_train_dev_sstat, X_train_dev_dft, y_train_dev)
    predictions = prediction_model.predict(X_test_sstat, X_test_dft)
    f1_results.append(sklearn.metrics.f1_score(y_test, predictions))
    precision_results.append(sklearn.metrics.precision_score(y_test, predictions))
    recall_results.append(sklearn.metrics.recall_score(y_test, predictions))
  print('Equal-Weighted Ensemble achieved {} f1, {} precision, {} recall'.format(np.mean(f1_results), np.mean(precision_results), np.mean(recall_results)))
  print()

  #### Baseline -- all True
  f1_results = []
  precision_results = []
  recall_results = []
  for i in range(50):
    predictions = [1 for x in y_test]
    f1_results.append(sklearn.metrics.f1_score(y_test, predictions))
    precision_results.append(sklearn.metrics.precision_score(y_test, predictions))
    recall_results.append(sklearn.metrics.recall_score(y_test, predictions))
  print('All-True baseline achieved {} f1, {} precision, {} recall'.format(np.mean(f1_results), np.mean(precision_results), np.mean(recall_results)))
  print()

  #### Baseline -- random prediction
  f1_results = []
  precision_results = []
  recall_results = []
  for i in range(50):
    predictions = np.random.randint(0,2,y_test.size)
    f1_results.append(sklearn.metrics.f1_score(y_test, predictions))
    precision_results.append(sklearn.metrics.precision_score(y_test, predictions))
    recall_results.append(sklearn.metrics.recall_score(y_test, predictions))
  print('Coin flip baseline achieved {} f1, {} precision, {} recall'.format(np.mean(f1_results), np.mean(precision_results), np.mean(recall_results)))
  print()

  #### Baseline -- random 23% prediction
  f1_results = []
  precision_results = []
  recall_results = []
  for i in range(50):
    predictions = np.random.uniform(0,1, size=y_test.size) <= .2373
    f1_results.append(sklearn.metrics.f1_score(y_test, predictions))
    precision_results.append(sklearn.metrics.precision_score(y_test, predictions))
    recall_results.append(sklearn.metrics.recall_score(y_test, predictions))
  print('Weighted-coin flip baseline achieved {} f1, {} precision, {} recall'.format(np.mean(f1_results), np.mean(precision_results), np.mean(recall_results)))
  print()


def gad7_final_dev(args, prespecified_model
    , X_train_dev_sstat, X_train_dev_dft, y_train_dev):
  '''
    Trains/evaluates models to predict the GAD7 variable from head movement data.
    Performs hold-one-out cross-validation.

    Trains models on both DFT and summary statistic data.
  '''
  #tracking_data = '../data/Tracking/'
  #part_data = '../data/participant_data.csv'

  ## load usable (pid, mo) pairs, and make sure to remove test set
  #pid_mos_sg = util.which_parts_have_score(part_data, util.gad7)
  #pid_mos_t = util.which_parts_have_tracking_data(tracking_data)
  #pid_mos_both = list(set(pid_mos_sg) & set(pid_mos_t))
  #pid_mos_use = list(filter(lambda pm : pm[0].upper() not in test_participants, pid_mos_both))
  ##pid_mos_use = list(filter(lambda pm: pm[1] == 0, pid_mos_use))
  #print('Loaded {} (pid, mo) pairs with both tracking data and GAD7 scores.'.format(len(pid_mos_both)))
  #print('Removed {} (pid, mo) test set pairs to leave {} total to train with.'.format(len(pid_mos_both) - len(pid_mos_use), len(pid_mos_use)))

  #X_train_dev_sstat = util.compute_fvecs_for_parts(pid_mos_use, 'summary_stat')#[:,1:10]
  #X_train_dev_dft = util.compute_fvecs_for_parts(pid_mos_use, 'dft')#[:,1:10]
  #scores = util.load_scores(part_data, pid_mos_use, util.gad7)
  #y_train_dev = np.array(scores) >= 10

  kf = KFold(n_splits=118, random_state=1)
  kf_train_avg_errors = []
  kf_val_errors = []
  predictions = []
  for train_index, test_index in kf.split(X_train_dev_sstat):
    X_train_sstat, X_dev_sstat = X_train_dev_sstat[train_index], X_train_dev_sstat[test_index]
    X_train_dft, X_dev_dft = X_train_dev_dft[train_index], X_train_dev_dft[test_index]
    y_train, y_dev = y_train_dev[train_index], y_train_dev[test_index]

    # Train model
    if prespecified_model:
      gad_model = prespecified_model
    else:
      gad_model = models.ClassificationGAD7Model()
    gad_model.fit(X_train_sstat, X_train_dft, y_train)

    y_train_predict = gad_model.predict(X_train_sstat, X_train_dft ) > .5
    y_train_error = np.mean(np.abs(y_train ^ y_train_predict))
    kf_train_avg_errors.append(y_train_error)

    # Predict on the held-out example
    y_predict = gad_model.predict(X_dev_sstat, X_dev_dft)
    predictions.append(y_predict[0])
    y_val_error = np.mean(np.abs(y_predict ^ y_dev))
    kf_val_errors.append(y_val_error)
    #print("\nModel predictions: {} \nTrue labels:       {} \n".format(y_predict, y_hold_out))
  #print("Avg GAD7 train error: {}".format(np.mean(kf_train_avg_errors)))
  print()
  model_f1 = sklearn.metrics.f1_score(y_train_dev, predictions)
  model_precision = sklearn.metrics.precision_score(y_train_dev, predictions)
  model_recall = sklearn.metrics.recall_score(y_train_dev, predictions)
  print("GAD7 F1: {}".format(model_f1))
  print("GAD7 prec: {}".format(model_precision))
  print("GAD7 recall: {}".format(model_recall))
  #print(y_train_dev)
  #print(predictions)
  #print("Always predict false: {}".format(np.mean(np.abs(y_train_dev ^ mode(y_train_dev)[0]))))
  print()
  always_predict_true = [1 for x in y_train_dev]
  always_true_f1 = sklearn.metrics.f1_score(y_train_dev, always_predict_true)
  always_true_precision = sklearn.metrics.precision_score(y_train_dev, always_predict_true)
  always_true_recall = sklearn.metrics.recall_score(y_train_dev, always_predict_true)
  print("Always predict true f1: {}".format(always_true_f1))
  print("Always predict true prec: {}".format(always_true_precision))
  print("Always predict true recall: {}".format(always_true_recall))
  print()
  random_predict = np.random.randint(0,2,y_train_dev.size)
  random_true_f1 = sklearn.metrics.f1_score(y_train_dev, random_predict)
  random_true_precision = sklearn.metrics.precision_score(y_train_dev, random_predict)
  random_true_recall = sklearn.metrics.recall_score(y_train_dev, random_predict)
  print("Predict randomly f1: {}".format(random_true_f1))
  print("Predict randomly prec: {}".format(random_true_precision))
  print("Predict randomly recall: {}".format(random_true_recall))
  print()
  print()
  random_20_percent = np.random.uniform(0,1, size=y_train_dev.size) <= .2373
  random_20_f1 = sklearn.metrics.f1_score(y_train_dev, random_20_percent)
  random_20_precision = sklearn.metrics.precision_score(y_train_dev, random_20_percent)
  random_20_recall = sklearn.metrics.recall_score(y_train_dev, random_20_percent)
  print("Predict random 20% have anxiety f1: {}".format(random_20_f1))
  print("Predict random 20% have anxiety prec: {}".format(random_20_precision))
  print("Predict random 20% have anxiety recall: {}".format(random_20_recall))
  print()
  print(np.mean(y_train_dev))
  #print("Avg GAD7 kfold val: {}".format(np.mean(kf_val_errors)))
  results = {
      'model_f1': model_f1,
      'model_precision': model_precision,
      'model_recall': model_recall,
      'always_true_f1': always_true_f1,
      'always_true_precision': always_true_precision,
      'always_true_recall': always_true_recall,
      'random_true_f1': random_true_f1,
      'random_true_precision':random_true_precision,
      'random_true_recall':random_true_recall,
      'random_20_f1':random_20_f1,
      'random_20_precision':random_20_precision,
      'random_20_recall':random_20_recall
      }
  return results


def gad7_kfold_classification(args, prespecified_model=None):
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
      print(train_index, test_index)
      X_train, X_dev = X_train_dev[train_index], X_train_dev[test_index]
      y_train, y_dev = y_train_dev[train_index], y_train_dev[test_index]

      # Train model
      if prespecified_model:
        gad_model = prespecified_model
      else:
        gad_model = models.ClassificationGAD7Model()
      gad_model.fit(X_train, y_train)

      y_train_predict = gad_model.predict(X_train) > .5
      y_train_error = np.mean(np.abs(y_train ^ y_train_predict))
      kf_train_avg_errors.append(y_train_error)

      # Predict on the held-out example
      y_predict = gad_model.predict(X_dev)
      predictions.append(y_predict[0])
      for index, (y_one, yhat_one) in enumerate(zip(y_dev, y_predict)):
        if y_one != y_predict and  y_one == 1:
          print('wrong', test_index, scores[test_index[0]])
      for index, (y_one, yhat_one) in enumerate(zip(y_dev, y_predict)):
        if y_one == y_predict and y_one  == 1:
          print('right', test_index, scores[test_index[0]])

      y_val_error = np.mean(np.abs(y_predict ^ y_dev))
      kf_val_errors.append(y_val_error)
      #print("\nModel predictions: {} \nTrue labels:       {} \n".format(y_predict, y_hold_out))
    #print("Avg GAD7 train error: {}".format(np.mean(kf_train_avg_errors)))
    print()
    model_f1 = sklearn.metrics.f1_score(y_train_dev, predictions)
    model_precision = sklearn.metrics.precision_score(y_train_dev, predictions)
    model_recall = sklearn.metrics.recall_score(y_train_dev, predictions)
    print("GAD7 F1: {}".format(model_f1))
    print("GAD7 prec: {}".format(model_precision))
    print("GAD7 recall: {}".format(model_recall))
    #print(y_train_dev)
    #print(predictions)
    #print("Always predict false: {}".format(np.mean(np.abs(y_train_dev ^ mode(y_train_dev)[0]))))
    print()
    always_predict_true = [1 for x in y_train_dev]
    always_true_f1 = sklearn.metrics.f1_score(y_train_dev, always_predict_true)
    always_true_precision = sklearn.metrics.precision_score(y_train_dev, always_predict_true)
    always_true_recall = sklearn.metrics.recall_score(y_train_dev, always_predict_true)
    print("Always predict true f1: {}".format(always_true_f1))
    print("Always predict true prec: {}".format(always_true_precision))
    print("Always predict true recall: {}".format(always_true_recall))
    print()
    random_predict = np.random.randint(0,2,y_train_dev.size)
    random_true_f1 = sklearn.metrics.f1_score(y_train_dev, random_predict)
    random_true_precision = sklearn.metrics.precision_score(y_train_dev, random_predict)
    random_true_recall = sklearn.metrics.recall_score(y_train_dev, random_predict)
    print("Predict randomly f1: {}".format(random_true_f1))
    print("Predict randomly prec: {}".format(random_true_precision))
    print("Predict randomly recall: {}".format(random_true_recall))
    print()
    print()
    random_20_percent = np.random.uniform(0,1, size=y_train_dev.size) <= .2373
    random_20_f1 = sklearn.metrics.f1_score(y_train_dev, random_20_percent)
    random_20_precision = sklearn.metrics.precision_score(y_train_dev, random_20_percent)
    random_20_recall = sklearn.metrics.recall_score(y_train_dev, random_20_percent)
    print("Predict random 20% have anxiety f1: {}".format(random_20_f1))
    print("Predict random 20% have anxiety prec: {}".format(random_20_precision))
    print("Predict random 20% have anxiety recall: {}".format(random_20_recall))
    print()
    print(np.mean(y_train_dev))
    #print("Avg GAD7 kfold val: {}".format(np.mean(kf_val_errors)))
    results = {
        'model_f1': model_f1,
        'model_precision': model_precision,
        'model_recall': model_recall,
        'always_true_f1': always_true_f1,
        'always_true_precision': always_true_precision,
        'always_true_recall': always_true_recall,
        'random_true_f1': random_true_f1,
        'random_true_precision':random_true_precision,
        'random_true_recall':random_true_recall,
        'random_20_f1':random_20_f1,
        'random_20_precision':random_20_precision,
        'random_20_recall':random_20_recall
        }
    return results

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

    X_train_dev = util.compute_fvecs_for_parts(pid_mos_use, args.featurization)[:,1:2]
    scores = util.load_scores(part_data, pid_mos_use, util.gad7)
    y_train_dev = np.array(scores) 

    kf = KFold(n_splits=118, random_state=1)
    kf_train_avg_errors = []
    kf_val_errors = []
    for i in range(10):
      for train_index, test_index in kf.split(X_train_dev):
        X_train, X_dev = X_train_dev[train_index], X_train_dev[test_index]
        y_train, y_dev = y_train_dev[train_index], y_train_dev[test_index]

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

    X_train_dev = util.compute_fvecs_for_parts(pid_mos_use, 'dft')
    scores = util.load_scores(part_data, pid_mos_use, util.gad7)
    y_train_dev = np.array(scores) 
    for index, y in enumerate(y_train_dev):
      print(index, y)

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
    palette = itertools.cycle(sns.color_palette())
    color = next(palette)
    color = next(palette)
    color = next(palette)
    color1 = next(palette)
    color2 = next(palette)

    # Construct graphs of one single high-anxiety and one single low-anxiety participant.
    high_path = '../data/Tracking/TRACKING_MV08176R.TXT'
    high_val = 15

    low_path  = '../data/Tracking/TRACKING_MV0764712MOR.TXT'
    low_val = 2

    ####Error analysis
    ##low_path = '../data/all_tracking/tracking_PA201476MOR.txt' # Wrong, and was 1
    ##low_path = '../data/all_tracking/tracking_PA3010412MOR.txt' # Wrong, nad was 1
    ##low_path = 'tmp.txt' # Wrong, and was 1
    #low_path = '../data/all_tracking/tracking_PA25084R.txt' # Wrong, and was 1
    #low_val = 0
    #high_val = 1  # Right, and was 1
    ##high_path = '../data/all_tracking/tracking_MV09586R.txt'
    #high_path = 'tmp2.txt' # right, and was 1
    #high_path = '../data/all_tracking/tracking_PA26203R.txt' # Right, and was 1
    ## GAD& 15
    ####END Error analysis

    high_lines = [x.split('\t') for x in tqdm(open(high_path))]
    #for x in high_lines:
    #  print('aah')
    #  print((x[0], json.loads(x[1]), json.loads(x[2])))
    #  print('aah')
    high_lines = [(x[0], json.loads(x[1]), json.loads(x[2])) for x in high_lines]
    low_lines = [x.split('\t') for x in tqdm(open(low_path))]
    #print('low')
    #for x in open(low_path):
    #  print('aah')
    #  print((x[0], json.loads(x[1]), json.loads(x[2])))
    #  print('aah')
    low_lines = [(x[0], json.loads(x[1]), json.loads(x[2])) for x in tqdm(low_lines)]

    fig, axes = plt.subplots(3,2, figsize=(15,6))

    channel_names = ['Left Channel', 'Right Channel']
    axis_names = ['Pitch', 'Roll', 'Yaw']

    for axis_index in [0,1,2]:
      for channel_index in [0,1]:
        axes[axis_index][channel_index].scatter(range(len(high_lines)), [x[channel_index+1][axis_index] for x in high_lines], label='High-Anxiety Participant, GAD7=15', color=color1)
        axes[axis_index][channel_index].scatter(range(len(low_lines)), [x[channel_index+1][axis_index] for x in low_lines], label='Low-Anxiety Participant, GAD7=2', color=color2)
        #axes[axis_index][channel_index].scatter(range(len(high_lines)), [x[channel_index+1][axis_index] for x in high_lines], label='High-Anxiety Participant, Predicted Correctly')
        #axes[axis_index][channel_index].scatter(range(len(low_lines)), [x[channel_index+1][axis_index] for x in low_lines], label='High-Anxiety Participant, Predicted Incorrectly')
        axes[axis_index][channel_index].set_title(channel_names[channel_index] + ' ' + axis_names[axis_index], fontsize=17)
    plt.tight_layout()
    plt.suptitle('Raw Head Movement Data for Sample High-GAD7 and Low-GAD7 Participants')
    #plt.suptitle('Error analysis: Raw Head Movement data for High-Anxiety Patients with Correct and Incorrect Prediction', fontsize=18)
    plt.legend(fontsize=12)
    plt.subplots_adjust(top=0.85)
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
    X_train_dev = util.compute_fvecs_for_parts(pid_mos_use, 'dft')
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

