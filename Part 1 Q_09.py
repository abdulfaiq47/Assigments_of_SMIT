"""PART 01
Q9.	Write a function to take user's score as argument and determine if they pass or fail (pass if score is above 60, otherwise fails.)

"""
def result():
    marks=float(input("Enter you scrore: ")) 

    if marks >= 60:
        print("You passed the exam.")
    else:
        print("you failed the exam.")    
        

result()       