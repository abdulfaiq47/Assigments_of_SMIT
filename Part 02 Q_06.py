"""PART 02
Q6.	From the given list:
Gadgets = [“Mobile”, “Laptop”, 10.0, “Marker”, 3, 10, “Speakers”, “Camera”, 310.25]
a.	Create separate list of string and numbers,
b.	Sort the string list in ascending and descending order
c.	Sort the string list by length of elements in the list
d.	Sort the numbers list in ascending and descending order

"""
Gadgets = ["Mobile", "Laptop", 10.0, "Marker", 3, 10, "Speakers", "Camera", 310.25]

# a.	Create separate list of string and numbers,
str_list =[] #create an empty list for strings
num_list=[] #create an empty list for numbers

for i in Gadgets:
    if type(i) == str:
        str_list.append(i)
    elif type(i) in [int, float]:  # check int and float types if true append
        num_list.append(i)  

#b.	Sort the string list in ascending and descending order
## Sorting in ascending order aplphabetically

str_list.sort()
print("String list in ascending order:", str_list) 
## Sorting in descending order alphabetically
str_list.sort(reverse=True) 
print("String list in descending order:", str_list)

# c.	Sort the string list by length of elements in the list
str_list.sort(key=len)  # Sort by length (shortest → longest)
print("String list sorted by length:", str_list)


#d.	Sort the numbers list in ascending and descending order

num_list.sort()  # Sort numbers in ascending order
print("Numbers list in ascending order:", num_list) 
num_list.sort(reverse=True)  # Sort numbers in descending order
print("Numbers list in descending order:", num_list)