import pandas as pd
import numpy as np
import geopandas
import matplotlib.pyplot as plt
import arc as arc
from geodatasets import get_path
import aqi
from colour import Color
import os

features = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
target = 'stroke'

def clear():
  os.system('cls' if os.name == 'nt' else 'clear')

#some of the bmi values were NaN so grouping by age and gender can approximately fill in the NaN values with the mean of the age and gender group
def questions(count):
  clear()
  next = False
  while (next != True and count != 11):
    match count:
      case 1:
        gender = input("Enter gender \n1.Male\n2.Female: ")
        if(gender != "1" and gender != "2"):
          print("Invalid input")
        else:
          next = True
      case 2:
        age = int(input("Enter age: "))
        if(age < 0 or age > 100):
          print("Invalid input")
        else:
          next = True
      case 3:
        smoking_status = input("Enter smoking status \n1.Formely smoked \n2.Never smoked\n3.Smoke: ")
        if(smoking_status != "1" and smoking_status != "2" and smoking_status != "3"):
          print("Invalid input")
        else:
          next = True
      case 4:
        bmi = int(input("Enter BMI: "))
        next = True
      case 5:
        ever_married = input("Ever been married \n1.Yes \n2.No: ")
        if(ever_married != "1" and ever_married != "2"):
          print("Invalid Input")
        else:
          next = True
      case 6:
        hypertension = input("Have you had hypertenstion before \n1.Yes \n2.No: ")
        if(hypertension != "1" and hypertension != "2"):
          print("Invalid Input")
        else:
          next = True
      case 7:
        residence_type = input("Enter residence type \n1.Rural \n2.Urban: ")
        if(residence_type != "1" and residence_type != "2"):
          print("Invalid Input")
        else:
          next = True
      case 8:
        avg_glucose = int(input("Enter average glucose level: "))
        next = True
      case 9:
        heart_disease = input("Have you had heart disease before \n1.Yes\n2.No: ")
        if(heart_disease != "1" and heart_disease != "2"):
          print("Invalid Input")
        else:
          next = True
      case 10:
        work_type = input("Enter work type \n1.Self-employed \n2.Private \n3.Govt \n4.Never Worked before \n5.With Children: ")
        if(work_type != "1" and work_type != "2" and work_type != "3" and work_type != "4"):
          print("Invalid Input")
        else:
          next = True
  questions(count+1)




def fill_bmi(df):
  mean = df.groupby(['age','gender'])['bmi'].transform('mean')
  df['bmi'] = df['bmi'].combine_first(mean)
  return df



#questions(1)
data = pd.read_csv(("healthcare-dataset-stroke-data.csv"))
data = fill_bmi(data)
data = data.sample(n=len(data))
testData = data.sample(n=int(len(data)*0.2))
data = data[~data.index.isin(testData.index)]
data = data.drop('id', axis=1)
strokes = data.loc[data['stroke'] == 1]
strokes = strokes.groupby('age')
ages = []
count = []
for age, group in strokes:
  ages.append(age)
  count.append(len(group))
print(ages,count)
fig, ax = plt.subplots()

ax.bar(ages, count, width=1, edgecolor="white", linewidth=1)

ax.set(xlim=(0, 100), xticks=np.arange(1, 100),
       ylim=(0, 20), yticks=np.arange(1, 20))

plt.show()