import pandas as pd
import numpy as np
import pickle
import questions
features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
target = 'stroke'

#some of the bmi values were NaN so grouping by age and gender can approximately fill in the NaN values with the mean of the age and gender group

test = questions.questions(1)
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
  if type(unique[0]) == str:
    mapping = {value: count for count, value in enumerate(unique)}
    data[column] = data[column].map(mapping)
data = data.sample(n=len(data))
testData = data.sample(n=int(len(data) * 0.2))
trainData = data[~data.index.isin(testData.index)]
Y_train = trainData[target]
X_test = testData
Y_test = testData[target]
p = 1 / (1 + np.exp(-(np.mean(Y_train))))
pTest = 1 / (1 + np.exp(-(np.mean(Y_test))))
residualsTest = Y_test - pTest
X_test['residuals'] = residualsTest
max_depth = 5


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
    value = tree.value
    #print("Value",value)
    return value
  feature = tree.feature
  threshold = tree.threshold
  #print(feature,x[1][feature],threshold)
  if (x[1][feature] <= threshold):
    left = tree.left
    return predict(left, x)
  else:
    right = tree.right
    return predict(right, x)

def testing(tree,x):
  predictions = []
  for rows in x.iterrows():
    value = predict(tree, rows)
    predictedValue = p + (0.1 * value)
    if(predictedValue>p):
      predictedValue = 1
    else:
      predictedValue = 0
    predictions.append(predictedValue)
  return predictions


def print_tree(node, depth=0):
  if node.value is not None:
    print(f"{'|  ' * depth}Predict: {node.value}")
  else:
    print(f"{'|  ' * depth}{node.feature} <= {node.threshold}")
    print_tree(node.left, depth + 1)
    print(f"{'|  ' * depth}{node.feature} > {node.threshold}")
    print_tree(node.right, depth + 1)




with open("tree.pkl", "rb") as f:
  tree = pickle.load(f)
predictions = []
"""test_predictions = pd.Series(testing(tree,X_test),index=Y_test.index)
count = 0
Y_test = pd.Series(Y_test)
for i in range(len(Y_test)):
  if(Y_test.iloc[i] == test_predictions.iloc[i]):
    count +=1
    """
userValues = testing(tree,test)
if(userValues[0] == 1):
  print("Prediction is positive")
else:
  print("Prediction is negative")
  