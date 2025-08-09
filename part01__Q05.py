""" part 01
Q 05 Write a program to take user’s age as input and display whether he is minor,
adult or senior citizen: a. Minor age is less than 18. b. Adult age is
greater than 18 and less than 60 c.Senior citizen age is greater than 60

"""

age = int(input("Enter your Age: "))

if age <= 17:
    print ("Minor age is less than 18.")
elif age >=18 and age <= 60:
    print("Adult age isgreater than 18 and less than 60")
else:
    print("Senior citizen age is greater than 60")
