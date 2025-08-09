""" PART :01
Q: 06 Write a program to take integer as input and check if it is even or odd.

"""
value = int(input("Enter the value: "))

"""  this lines  lgoics is when user enter value if value reminder
is 0 then is EVEN Example 2 / 2 and its reminder is 0 and in py we can find
reminder by using this '%'  """
if value % 2 == 0:
    print("Even")
else :
    print("Odd")
