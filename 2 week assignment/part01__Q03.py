""" PART : 01
Q:03 Write a program to find the largest of the three numbers

"""

x=int(input("Enter the value: "))
y=int(input("Enter the value: "))
z=int(input("Enter the value: "))

if x > y and x > z:
    print(f"The value you first value:{x} is greater then second and third val ")
elif y > z and y > x :
    print(f"The value you second value:{y} is greater then first and third val ")
else:
    print(f"The value you third value:{z} is greater then first and second val ")

    
