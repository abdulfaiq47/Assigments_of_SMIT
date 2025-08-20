"""Part-2: Data Structures (List)
Q2.	Write a code to separate strings, numbers and Boolean data from the above list into separate lists.


"""

list= ["Abdul Faiq",123,True] # create list 

for i in list: # get all index elemnts one by one 
    if  type(i) == str: # check type of 'i' is str or not by  
        print(f"This is String: {i}")
    elif  type(i) == int:
        print(f"This is Number: {i}")
    elif  type(i) == bool:   
        print(f"This is Boolean Data: {i}")
              