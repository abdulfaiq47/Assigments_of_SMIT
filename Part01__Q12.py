"""PART: 01
Q:12 Write a program that takes a temperature in Celsius and checks if it is
freezing, moderate or hot. a. Freezing temperature is below 0.
b. Moderate temperature is
greater than 0 and less than 26. c. Hot temperature is above 26."""

temp = int(input("ENTER THE TEMPERATURE"))

if temp <0:
    print("Freezing temperature is below 0.")
elif temp>0 and temp<26:
    print("Moderate temperature isgreater than 0 and less than 26")
else:
    print("Hot temperature is above 26")
