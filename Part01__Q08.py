""" PART: 01
Q:08 Write a program to take number as input and check if it lies in the
range i.e. 20(inclusive)– 40 (exclusive)


IN this question maybe i take value form user if he right between 20 and 40
print user lies in the range otherwise user in not in range 

"""


x= int(input("Enter the value: "))

if x>=20 and x<=40:
    print("The user lies in this range")
else:
    print("The user outside the range")
