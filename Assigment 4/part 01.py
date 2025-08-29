# Part 1: SRMS using nested list

# Write a menu-driven program to create a Student Records Management System (SRMS) using List and its methods. Your program must include the following features:
# a. Add Student – Take name, roll number, subject and marks as input and store them into a list.
# b. Display students – show all stored students records in a readable format.
# c. Search Student – Search for a student by roll number and display if found.
# d. Update record – Update the marks of student using roll number.
# e. Delete Record – Delete a student record using roll number.
# f. Sort Record – Sorting students records by marks (descending order).
# g. Exit.



# for Storing Students Records
Records = []

# ADD Students
def add_students():
    name = input("Enter your Name: ")
    Roll_no= input("Enter your Roll Number: ")
    Cousre= input("Enter your Course: ")
    Marks=int(input("Enter Your Marks: "))

    Records.append([name, Roll_no, Cousre, Marks])
    print("Record Enter Sucessfully...")

#2 : Display Students Data

def display_data():
    if Records is not None:
        for record in Records:
            print("----------------")
            print(f"Name:     {record[0]}")
            print(f"Roll No.: {record[1]}")
            print(f"Course:   {record[2]}")
            print(f"Marks:    {record[3]}")
    else:
        print("empty Record")

#3 : Search Students
        
def search_student():
##    if records:
##        roll_no= input("Enter your Roll Number To Search: ")
##        found = False
##        for record in records:
##            if roll_no == record[1]:
##                print(f"Name:     {record[0]}")
##                print(f"Roll No.: {record[1]}")
##                print(f"Course:   {record[2]}")
##                print(f"Marks:    {record[3]}")
##                found= True
##
##        if not found:
##            print("Record Not Found!!!")
##    else:
##        print("Empty Records")
    if Records:
        roll=input("Enter Roll Number To search The Student")
        for index, found in enumerate(Records):
            if found[1] == roll:
                
                
                return found, index
            
        return None, None    
    else:
        print("Empty Record")

#4 updates Marks

def update_marks():
    found, idx = search_student()
    if found is not None:
        print("Record Found")
        print("------------")
        print(f"Name:     {found[0]}")
        print(f"Roll No.: {found[1]}")
        print(f"Course:   {found[2]}")
        print(f"Marks:    {found[3]}")
        print("------------")

        update_markss = int(input("Enter Marks To update: "))

        #update
        Records[idx][3]= update_markss
        print("Marks updated Sucessfully.")
    else:
         print("Student Not Found ! Unable to update")
        
    

        

# 5 Delete
def delete():
    found, idx= search_student()
    if found is not None:
        Records.remove(found)
        print("Record Deleted Sucessfully....")

    else:
        print("Uable to Delete ! Record Not Found")
        
# sort
def sort():
    if Records:
        Records.sort(key=lambda x:x[3], reverse=True)
        print("Record Sorted..")

    else:
        print("Uable To sort ! empty")
    
while True:
    print("================")
    print("Type No. To perform Task")
    print("1 --> Add Students")
    print("2 --> Display Students Data")
    print("3 --> Search Student")
    print("4 --> Update Student")
    print("5 --> Delete Students")
    print("6 --> Sort Students --> By Marks")
    print("7 --> Exit")
    print("================")

    Choice = input("Enter YOur choice: ")

    if Choice == "1":
        add_students()
    elif Choice == "2":
        display_data()
    elif Choice == "3":
        student, _= search_student()
        if student:
            print("----------------")
            print(f"Name:     {student[0]}")
            print(f"Roll No.: {student[1]}")
            print(f"Course:   {student[2]}")
            print(f"Marks:    {student[3]}")
            print("----------------")

        else:
            print("Record Not Found!!!")   
    elif Choice == "4":
        update_marks()
    elif Choice == "5":
        delete()
    elif Choice == "6":
        sort()
    elif Choice == "7":
        break
    else:
        print("Invailed Enter")
    
