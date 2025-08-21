"""PART 1
Q8.	Write a function to compute the area and circumference of the circle and return the computed results.

The formula of Area od Circle is A = π * r²
The formula of Circumference of Circle is C = 2 * π * r 
here π = 3.14 , r is Radius 

"""

def circle_proper():
    radius = float(input("Enter The Radius of Circle: "))
    print(f"The Area of cirlce is : {3.14 * radius ** 2}\n"
          f"The Circumference of Circle is {2 * 3.14 * radius}")
    
circle_proper() 