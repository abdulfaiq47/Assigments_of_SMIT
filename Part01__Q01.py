""" PART : 01
Q:01 Write a program that checks if a given number is positive, negative or zero.
"""

x =int(input("Enter A value: "))

if x <0: #this say if x less than 0 where x is the stored value by user given
    print("Negative")
elif x > 0:
    print("Positive") #this say if x greater than 0
else: # this line if both above condition got failed than it zero 
    print("zero")
