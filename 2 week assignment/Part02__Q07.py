"""Part-2: Loops
Q:07 Write a program to compute the factorial of a number using while loop.

"""
# NOte this code is ai generated 
x=int(input("Enter a number to get factorial of a number: "))
factorial = 1
i=1

if x < 0:
    print(f"this not factorial {x} because the factorail is negative ")
elif x== 0:
    print(f"the factorail of {x} is '1'.")
else:
    while i <=x: # these three line logic i dont understand
        factorial *= i
        i += 1

print(f"The factorial of {x} is {factorial}")
