"""Part 1: Function
Q2.Write a function that takes a number as argument and check
if a given number is positive, negative or zero.

"""


def num_check():
    num=int(input("Enter Any Number: "))
    if num == 0:
        print("The Entered NUM is 'Zero'.")
    elif num <0 :
        print("The Entered NUM is 'Negative'.")
    else:
        print("The Entered NUM is 'Positive'.")

num_check()        
    
