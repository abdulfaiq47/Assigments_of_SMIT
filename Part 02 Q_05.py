"""Part-2: Data Structures (List)
Q5.	Create a list [32,54,66,11,77,10,90]
a.	Remove items from the list with values less than 20.
b.	Sort the list in ascending and descending order.
c.	Now, compute the average value of the list items.

"""
list = [32,54,66,11,77,10,90]
# a.	Remove items from the list with values less than 20.
for i in list:
    if i < 20:
        list.remove(i)
        print("Removed int less than 20", list)
    else:
        print("Int greater than 20", list) 

# b.  Sort the list in ascending and descending order.   
list.sort()
print("List in ascending order:", list)
list.sort(reverse=True)
print("List in descending order:", list)

# c.	Now, compute the average value of the list items.

average = sum(list) / len(list)  # here we calculate the average the sum of the list and divide it by the length of the list
print("Average value of the list items:", average)
# Note: The average is computed after removing items less than 20.
