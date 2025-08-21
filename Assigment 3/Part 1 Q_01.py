""" Part 1: Function 
Q1.Write a function named “Greetings” that takes user’s name and print greeting.
Output:       Welcome to SMIT training center, Ahsan 

"""

def greetings(): # make function named greeting
    name=input("Enter your Name: ") # take input from user must be str
    print(f"Welcome to SMIT training center," ,name) # the user entered name print the name where 'name' is define


greetings()        
