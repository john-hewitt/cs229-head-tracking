'''
Script for reading and reporting the results of the
random search over expert weights and threshold.

Reads results jsonl files and reports the set of weights,
and threshold, that resulted in the best Prec, Recall, and F1.
'''

import json
import sys
import argparse


argp = argparse.ArgumentParser()
argp.add_argument('report', help='Jsonlines report containing random search results')
args = argp.parse_args()


results_dict = {}
for line in open(args.report):
  report_line = json.loads(line.strip())
  weights = tuple(report_line[0])
  threshold = report_line[1]
  results = report_line[2]
  results_dict[(weights, threshold)] = results
  print('model_f1' in results)




sorted_by_precision = [(item, results_dict[item]) for item in sorted(results_dict.keys(), key = lambda x: -results_dict[x]['model_precision'])]
sorted_by_recall = [(item, results_dict[item]) for item in sorted(results_dict.keys(), key = lambda x: -results_dict[x]['model_recall'])]
sorted_by_f1 = [(item, results_dict[item]) for item in sorted(results_dict.keys(), key = lambda x: -results_dict[x]['model_f1'])]

print('Precision')
for record in zip(sorted_by_precision, range(5)):
  print(record)
print()
print('Recall')
for record in zip(sorted_by_recall, range(5)):
  print(record)
print()
print('F1')
for record in zip(sorted_by_f1, range(5)):
  print(record)






