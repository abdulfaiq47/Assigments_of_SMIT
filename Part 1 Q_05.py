"""PART: 01 
Q5.	Write a function to take user’s age as argument and return the message from the function whether he is minor, adult or senior citizen:
a.	Minor age is less than 18.
b.	Adult age is greater than 18 and less than 60
c.	Senior citizen age is greater than 60

"""

def check_age(age):
    
    if age < 18:
        return f"Minor age is less than 18, Your Age: {age}"
    elif age > 18 and age < 60:
        return f"Adult age is greater than 18 and less than 60, Your Age: {age}"
    else:
        return f"Senior citizen age is greater than 60, Your Age: {age}"
    
print(check_age())     
