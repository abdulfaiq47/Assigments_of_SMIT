"""PART:01
Q:Write a program that ask for an integer number and checks if it is divisible
by 2, 3, or both.

is this questrion probably user give inpuy in the form of int and i check
is this int is divisible by 2.3 or both 
"""

x= int(input("Enter the value: "))

if x % 2:
    print(f"Yes, this value : {x} is divisble by two")
elif x % 3 :
    print(f"Yes, this value : {x} is divisble by three")
elif x % 2 +3 :
    print(f"Yes, this value : {x} is divisble by both")
else:
    print("this value you give isn't divisible by 2,3 or both")

    
