"""Part-2: Loops
Q:03 Write a program to print multiplication table of a given number.
"""

x=int(input("Enter a No. to get table of this No.: "))


for i in range(1,11):
    print(f"{x} x {i} = ",x*i)
