""" PART 01
Q11.	Write a function to compute factorial of a given number using recursion technique.

"""                        

m= 7

def factorial(m):
    if m > 0:
        result = m * factorial(m -1)
        print(result)
    else:
        result = 1

    return result 

fac = factorial(m)

print(f"Factorial of {m} = {fac}")