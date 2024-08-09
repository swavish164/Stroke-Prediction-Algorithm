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
for column in data:
  unique = data[column].unique()
  count = 0
  if type(unique[0]) != int:
    mapping = {value: count for count, value in enumerate(unique)}
    data[column] = data[column].map(mapping)
data = data.sample(n=len(data))
testData = data.sample(n=int(len(data) * 0.2))
trainData = data[~data.index.isin(testData.index)]
X_train = trainData
Y_train = trainData[target]
X_test = testData
Y_test = testData[target]
initial_prediction = np.mean(Y_train)
p = 1 / (1 + np.exp(-initial_prediction))
residuals = Y_train - p
X_train['residuals'] = residuals
max_depth = 6


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


def split_and_count(grouped):
  size = len(grouped)
  midpoint = int(size / 2)
  if (midpoint > 0):
    lower = grouped[:midpoint]
    upper = grouped[midpoint:]
    splitPoint = grouped[midpoint]
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
  else:
    lower_stroke_count = 0
    upper_stroke_count = 0
    lowerCount = 1
    upperCount = 1
    splitPoint = 1
  return lower_stroke_count, upper_stroke_count, lowerCount, upperCount, splitPoint


def best_split(X):
  best_split = None
  best_score = -1
  grouped = []
  for feature in X.columns:
    if feature != 'stroke':
      groups = X.groupby(feature)
      grouped = []
      for group in groups:
        grouped.append(group)
    lower_counts, upper_counts, lowerCount, upperCount, splitPoint = split_and_count(grouped)
    giniLower = 1 - ((lower_counts / lowerCount)**2 +
                     ((lowerCount - lower_counts) / lowerCount)**2)
    giniUpper = 1 - ((upper_counts / upperCount)**2 +
                     ((upperCount - upper_counts) / upperCount)**2)
    gini = max(giniLower, giniUpper)
    if gini > best_score:
      best_score = gini
      best_feature = feature
      best_split = splitPoint
      lower = lowerCount
  return best_score, best_feature, best_split, lower


class DecisionTreeNode:

  def __init__(self,
               feature=None,
               threshold=None,
               left=None,
               right=None,
               value=None):
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.value = value

def predict(tree, x):
  if tree.value is not None:
    return tree.value
  feature = tree.feature
  threshold = tree.threshold
  print(threshold)
  if(x[feature] <= threshold):
    left = tree.left
    predict(tree,left)
  else:
    right = tree.right
    predict(tree,right)
    

def build_tree(X, maxDepth, depth=0):
  if (depth == max_depth):
    leaf_value = X['residuals'].mean()
    return DecisionTreeNode(value=leaf_value)
  gini, feature, split, middle = best_split(X)
  if feature is None:
    leaf_value = X['residuals'].mean()
    return DecisionTreeNode(value=leaf_value)
  featureGroup = X.sort_values(by=[feature])
  left_group = featureGroup[:middle]
  right_group = featureGroup[middle:]
  left_child = build_tree(left_group, max_depth, depth + 1)
  right_child = build_tree(right_group, max_depth, depth + 1)
  return DecisionTreeNode(feature=feature,
  threshold=featureGroup[:middle:][feature].iloc[0],
                          left=left_child,
                          right=right_child)


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
def gradientBoostingAlgorithm():
  pass


tree = build_tree(X_train, max_depth)
def print_tree(node, depth=0):
    """Recursively print the tree structure."""
    if node.value is not None:
        print(f"{'|  ' * depth}Predict: {node.value}")
    else:
        print(f"{'|  ' * depth}{node.feature} <= {node.threshold}")
        print_tree(node.left, depth + 1)
        print(f"{'|  ' * depth}{node.feature} > {node.threshold}")
        print_tree(node.right, depth + 1)

predict(tree, X_test)
print_tree(tree)
for rows in X_test.iterrows():
  value = predict(tree, rows)
  prediction = mean + 0.1 * int(value)