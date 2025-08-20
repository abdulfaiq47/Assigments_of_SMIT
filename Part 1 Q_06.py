"""PART: 01
Q6. Write a function to take integer as argument and
check if it is even or odd.

"""

def num_checker():
    num=int(input("Enter Any Number: "))
    if num % 2 == 0:
        print("even")
    else:
        print("odd")
num_checker()
