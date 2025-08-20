""" Part 1: Function 
Q:4 Write a function to take three numbers as argument
and return the largest number.

"""


def larger():
    num1=int(input("Enter Any Num1: "))
    num2=int(input("Enter Any Num2: "))
    num3=int(input("Enter Any Num3: "))

    if num1 > num2 and num1 > num3: # if num 1 is greater than num2 'and' means turn must be only one and num 1 is greater than num3 if ture run the bottom line under if 
        print("The Number you enterd first is Greater than second and third which is : ",num1)
    elif num2 > num1 and num2 > num3:
        print("The Number you enterd Second is Greater than first and third which is : ",num2)
    else:
        print("The Number you enterd third is Greater than first and second which is : ",num3)

larger()        
