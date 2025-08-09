""" PART : 01
Q:07 Write a program to take two numbers and an operator (/,x,+,-)
as input and perform the corresponding operation.
"""

# =======SIMPLE CALCULATOR========

x=float(input("Enter your first value: "))
y=input("Enter What's opertion you want to perform: ").lower()
#In 9 line .lower work if user input is 'ADd' this convert into 'add' to lowercase
z=float(input("Enter your second value: "))
#i use float because if user want in decimal working then this work

""" NOTE :for add type 'add'
          for subtract type 'subtract'
          for multiply type 'multiply'
          for divide type 'divide'
"""

if y=="add":
    print("This sum the given No.: ", x+z)
elif y=="subtract":
    print("This sub the given No.: ", x-z)
elif y=="multiply":
    print("This mul the given No.: ", x*z)
elif y=="divide":
    print("This div the given No.: ", x/z)    
