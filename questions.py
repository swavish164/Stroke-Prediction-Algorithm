import os
import pandas as pd


def clear():
  os.system('cls' if os.name == 'nt' else 'clear')


gender = 0
age = 0
hypertension = 0
heart_disease = 0
ever_married = 0
work_type = 0
residence_type = 0
avg_glucose = 0
bmi = 0
smoking_status = 0
data = pd.DataFrame()


def questions(count):
  global data
  clear()
  next = False
  while (next != True and count != 11):
    global gender
    global age
    global hypertension
    global heart_disease
    global ever_married
    global work_type
    global residence_type
    global avg_glucose
    global bmi
    global smoking_status
    match count:
      case 1:
        gender = input("Enter gender \n1.Male\n2.Female: ")
        if (gender != "1" and gender != "2"):
          print("Invalid input")
        else:
          next = True
      case 2:
        age = int(input("Enter age: "))
        if (age < 0 or age > 100):
          print("Invalid input")
        else:
          next = True
      case 3:
        smoking_status = input(
            "Enter smoking status \n1.Formely smoked \n2.Never smoked\n3.Smoke: "
        )
        if (smoking_status != "1" and smoking_status != "2"
            and smoking_status != "3"):
          print("Invalid input")
        else:
          next = True
      case 4:
        bmi = float(input("Enter BMI: "))
        next = True
      case 5:
        ever_married = input("Ever been married \n1.Yes \n2.No: ")
        if (ever_married != "1" and ever_married != "2"):
          print("Invalid Input")
        else:
          next = True
      case 6:
        hypertension = input(
            "Have you had hypertenstion before \n1.Yes \n2.No: ")
        if (hypertension != "1" and hypertension != "2"):
          print("Invalid Input")
        else:
          next = True
      case 7:
        residence_type = input("Enter residence type \n1.Rural \n2.Urban: ")
        if (residence_type != "1" and residence_type != "2"):
          print("Invalid Input")
        else:
          next = True
      case 8:
        avg_glucose = float(input("Enter average glucose level: "))
        next = True
      case 9:
        heart_disease = input(
            "Have you had heart disease before \n1.Yes\n2.No: ")
        if (heart_disease != "1" and heart_disease != "2"):
          print("Invalid Input")
        else:
          next = True
      case 10:
        work_type = input(
            "Enter work type \n1.Self-employed \n2.Private \n3.Govt \n4.Never Worked before \n5.With Children: "
        )
        if (work_type != "1" and work_type != "2" and work_type != "3"
            and work_type != "4"):
          print("Invalid Input")
        else:
          next = True
    if(next == True):
      clear()
      next = False
      count+=1
  data = data._append({'gender':int(gender), 'age':int(age), 'hypertension':int(hypertension), 'heart_disease':int(heart_disease),"ever_married":int(ever_married), 'work_type':int(work_type), 'residence_type':int(residence_type),'avg_glucose_level': avg_glucose,'bmi':bmi,'smoking_status':int(smoking_status)},ignore_index = True)
  return data