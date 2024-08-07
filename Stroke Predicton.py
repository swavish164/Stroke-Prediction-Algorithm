import pandas as pd
import numpy as np
import geopandas
import matplotlib.pyplot as plt
import arc as arc
from geodatasets import get_path
import aqi
from colour import Color
import os

features = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]
target = 'stroke'


def clear():
  os.system('cls' if os.name == 'nt' else 'clear')


#some of the bmi values were NaN so grouping by age and gender can approximately fill in the NaN values with the mean of the age and gender group


def fill_bmi(df):
  mean = df.groupby(['age', 'gender'])['bmi'].transform('mean')
  df['bmi'] = df['bmi'].combine_first(mean)
  return df


#questions(1)
data = pd.read_csv(("healthcare-dataset-stroke-data.csv"))
data = data[data['gender'] != 'Other']
data = data[data['age'] >= 16]
data = fill_bmi(data)
data = data.drop('id', axis=1)
data = data.sample(n=len(data))
testData = data.sample(n=int(len(data) * 0.2))
trainData = data[~data.index.isin(testData.index)]
X_train = trainData
Y_train = trainData[target]
X_test = testData
Y_test = testData[target]


class gradientBoostingRegressor:

  def __init__(
      self,
      learning_rate=0.1,
      n_estimators=100,
      max_depth=3,
  ):
    self.learning_rate = learning_rate
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.models = []
    self.losses = []

  def buildTree(self, X, Y, depth):
    if depth == self.max_depth:
      return
    #for i in range(len(X)):



def split_and_count(grouped):
  size = len(grouped)
  midpoint = int(size / 2)
  lower = grouped[:midpoint]
  upper = grouped[midpoint:]
  lowerCount = 0
  upperCount = 0
  upper_stroke_count = 0
  lower_stroke_count = 0
  for groups in lower:
    lowerCount += groups[1]['stroke'].size
    try:
      lower_stroke_count += groups[1]['stroke'].value_counts()[1]
    except KeyError:
      pass 
  for groups in upper:
    upperCount += groups[1]['stroke'].size
    try:
      upper_stroke_count += groups[1]['stroke'].value_counts()[1]
    except KeyError:
      pass
  return lower_stroke_count, upper_stroke_count,lowerCount, upperCount


def best_split(X, Y):
  best_split = None
  best_score = -1
  for feature in X.columns:
    if feature != 'stroke':
      groups = X.groupby(feature)
      grouped = []
      for group in groups:
        grouped.append(group)
      lower_counts, upper_counts,lowerCount,upperCount = split_and_count(grouped)
    giniLower = 1 - ((lower_counts / lowerCount)**2 + ((lowerCount-lower_counts)/lowerCount)**2 ) 
    giniUpper = 1 - ((upper_counts / upperCount)**2 + ((upperCount-upper_counts)/upperCount)**2 )
    gini = max(giniLower, giniUpper)
    print(gini)
    if gini > best_score:
      best_score = gini
      best_split = (feature, None)
  return best_score

"""
def gradientBoostingAlgorithm():
  initial_prediction = np.mean(Y_train)
  p = 1 / (1 + np.exp(-initial_prediction))
  residuals = Y_train - p
    for i in range(len(residuals)):
      residuals[i] = residuals[i] * p * (1 - p)
    gradient = np.sum(residuals) / len(residuals)
    self.models.append(prediction)
    self.losses.append(gradient)
    prediction = prediction + self.learning_rate * gradient
"""
best_split(X_train, Y_train)
