"""
Exercise from "data science from scratch", with some notes and debugging
so it is clearer what's going on

this is a more transparent bit of code showing how tree classifiers work
(by calculating entropy and assigning it to a partition to decide which
 questions we can ask about the data to filter down the dataset)
"""

from __future__ import division
from collections import Counter, defaultdict
import math,random

def entropy(class_probabilities):
  """given a list of class probabilities, compute the entropy
  	 - lower entropy means every value falls quite close to a class
  	 returns the entropy for the subset based on probabilities
  """
  return sum(-p * math.log(p,2) for p in class_probabilities if p)

def class_probabilities(labels):
  """ returns percentages of how many times each class occurs
  """
  total_count = len(labels)
  return [count / total_count for count in Counter(labels).values()]

def data_entropy(labeled_data):
  labels = [label for _, label in labeled_data]
  #as a percentage, how many in this subset lead to label True, how many lead to false?
  probabilities = class_probabilities(labels)
  #return the entropy associated with the labels for the subset
  return entropy(probabilities)

def partition_entropy(subsets):
  """ find the entropy from this partition of data into subsets
      subsets is a list of lists of labelled data 
  """
  total_count = sum(len(subset) for subset in subsets)
  #return the total entropy for the partition for all subsets in the partition
  return sum(data_entropy(subset) * len(subset) / total_count for subset in subsets)

def partition_by(inputs,attribute):
  """each input is a pair (attribute_dict, label) 
     return value is a dict where attribute_value -> inputs
  """
  groups = defaultdict(list)
  for input_pair in inputs:
  	key = input_pair[0][attribute]
  	groups[key].append(input_pair)
  #print len(groups)
  return groups

def partition_entropy_by(inputs,attribute):
  """ compute entropy corresponding to given partition """
  #sorts our dict data lines into groups, based on the attribute
  #we're sorting by
  partitions = partition_by(inputs,attribute)

  return partition_entropy(partitions.values())

if __name__ == "__main__":

  inputs = [
    ({'level':'Senior','lang':'Java','tweets':'no','phd':'no'},   False),
    ({'level':'Senior','lang':'Java','tweets':'no','phd':'yes'},  False),
    ({'level':'Mid','lang':'Python','tweets':'no','phd':'no'},     True),
    ({'level':'Junior','lang':'Python','tweets':'no','phd':'no'},  True),
    ({'level':'Junior','lang':'R','tweets':'yes','phd':'no'},      True),
    ({'level':'Junior','lang':'R','tweets':'yes','phd':'yes'},    False),
    ({'level':'Mid','lang':'R','tweets':'yes','phd':'yes'},        True),
    ({'level':'Senior','lang':'Python','tweets':'no','phd':'no'}, False),
    ({'level':'Senior','lang':'R','tweets':'yes','phd':'no'},      True),
    ({'level':'Junior','lang':'Python','tweets':'yes','phd':'no'}, True),
    ({'level':'Senior','lang':'Python','tweets':'yes','phd':'yes'},True),
    ({'level':'Mid','lang':'Python','tweets':'no','phd':'yes'},    True),
    ({'level':'Mid','lang':'Java','tweets':'yes','phd':'no'},      True),
    ({'level':'Junior','lang':'Python','tweets':'no','phd':'yes'},False)]
  
  for key in ['level','lang','tweets','phd']:
    print key,partition_entropy_by(inputs,key)
  print

  senior_inputs = [(input_value,label) for input_value,label in inputs if input_value['level'] == 'Senior']

  for key in ['lang','tweets','phd']:
  	print key,partition_entropy_by(senior_inputs,key)