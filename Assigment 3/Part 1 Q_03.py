"""Part 1: Function
Q3. Write a function to take two numbers as arguments
and return the larger number.

"""

def larger():
    input_1=float(input("Enter Any Number: "))
    input_2=float(input("Enter Any Number: "))

    if input_1 > input_2:
        print("The Number You Entered First is Greater than Second.")
    else:
        print("The Number You Entered Second is Greater than first.")
        
larger()
