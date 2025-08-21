"""Part-2: Data Structures (List)
Q4.	Write a code to get first and second best scores from the list:
Scores_list = [40, 89, 90, 89, 23, 90, 50]

"""
Scores_list = [40, 89, 90, 89, 23, 90, 50]
Scores_list.sort(reverse=True) # this will sort the list in ascending order for better readability or result and here we are using reverse=True to sort in descending order for best scores to be on top.
print("Sorted list in ascending order:", Scores_list)

for i in Scores_list:
    if i >=90 :
        print(f"best scores is : {i}")
    elif i >= 89:
        print(f"second best scores is : {i}")  
    elif i >= 50:
        print(f"third best scores is : {i}")          
    elif i >= 40:
        print(f"fourth best scores is : {i}")  
    else:
        print(f"fifth best scores is : {i}")  
        