"""Part-2: Data Structures (List)
Q3.	Create a list [“apple”, “raspberry”, “pineapple”, “cherry”]. 
a.	 How can you check if apple is present in the list and get the index of the element (if present)
b.	Now replace the raspberry and pineapple with orange.
c.	Insert “apricot” at index 2.
d.	Extend the resulting list with items “car”, “bike”, “aeroplane”.

"""

list = ["apple", "raspberry", "pineapple", "cherry"]

# a. Check if "apple" is present and get its index
if "apple" in list:
    print(f"Yes, 'Apple' is in the list and the Index of apple is:  '{list.index("apple")}'")

# b. Replace "raspberry" and "pineapple" with "orange"
print("===replace raspberry and pineapple with orange===")  
del list[1:3] # Remove "raspberry" and   "pineapple"
list.append("orange")  # Insert "orange" 
print(list)

# c.	Insert “apricot” at index 2.
print("==insert apricot at index 2===")
list.insert(2,"apricot")
print(list)

# d.	Extend the resulting list with items “car”, “bike”, “aeroplane”.
print("===extend the list with car, bike, aeroplane==")
list.extend(["car","bike","areoplane"])
print(list)


