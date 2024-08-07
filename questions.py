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
