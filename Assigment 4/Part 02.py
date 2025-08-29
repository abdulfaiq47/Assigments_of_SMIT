## Part 02 Student Record Management System (SRMS) using list of Dictionaries

# Write a menu-driven program to create a Student Records Management System (SRMS) using List of Disctionaires  and its methods. 
# Your program must include the following features:

# a. Add Student – Take name, roll number, subject and marks as input and store them into a list.
# b. Display students – Show all stored students records in a readable format.
# c. Search Student – Search for a student by roll number and display if found.
# d. Update record – Update the marks of student using roll number.
# e. Delete Record – Delete a student record using roll number.
# f. Sort Record – Sorting students records by marks (descending order).
# g. Exit.



records=[] # create an empty  'List of Disctionaires'  to store student records

# 1:  Add Student – Take name, roll number, subject and marks as input and store them into a list.

def add_student(): #make a function to add student
    print("------------------------") # get value form user 
    name=input("Enter Student Name: ")
    roll_no=input("Enter Roll No.: ")
    course=input("Enter Course Name: ")
    marks=int(input("Enter Marks: "))

    records.append({  #  in line number 16  i created empty list of dictionary now .append use to add the value and in above i get value from user store in 'Nested dictionary' and this use ({}) for Nested dictionary 
        "name": name, # assign the value to key like this name = name
        "roll_no": roll_no,
        "course": course,
        "marks": marks
    })

 
    print("------------------------")
    print("Student Added Successfully!!!")
# 2. Display students – Show all stored students records in a readable format.
def display_students():
    if records:   # check if record is not empty
        for student in records: # use for loop to get student from records
            print("------------------------")
            print(f"Name:     {student['name']}")
            print(f"Roll No.: {student['roll_no']}")
            print(f"Course:   {student['course']}")
            print(f"Marks:    {student['marks']}")
            print("------------------------")
    else:
        print("No Records Found!!!")        



# 3. Search Student – Search for a student by roll number and display if found.

def search_student():   # make a function to search student
    roll_no=input("Enter Roll No. to Search: ")    # get roll no to search
    if records:# if record is not empty
        for index, student in enumerate(records): # use for loop to get index and student by using enumerate built in function  'enumerate() is used to get index and value '
            if roll_no == student["roll_no"]: # check if roll no is equal to student roll no
                return student, index # if found return student and index
        return None, None    # if not found return None, None "index = none student = none"
    else:
        print("No Records Found!!!")

# 4. Update record – Update the marks of student using roll number.
def update_marks(): 
    student, index= search_student()  # in line 59 the returned saved in 'student' and 'index' 
    if student:  # if student is not None
            print("----------------")
            print(f"Name:     {student["name"]}")
            print(f"Roll No.: {student["roll_no"]}")
            print(f"Course:   {student["course"]}")
            print(f"Marks:    {student["marks"]}")
            print("----------------")
            update_marks=int(input("Enter New Marks: ")) # get new marks to update

            records[index]["marks"]= update_marks # reord is list of dic '[index]'  is the index of student and ["marks"] is marks of students = update_marks to update "records[0][78]= 89 like that"
            print("----------------")

            print("Record Successfully Updated!!!")  
            print("----------------")     

    else:
        print("----------------") 
        print("unable to update !! Record Not Found!!!")

# 5. Delete Record – Delete a student record using roll number. 

def delete():
    student, index= search_student()
    if student:  # if student is not None
            print("----------------")
            print(f"Name:     {student["name"]}")
            print(f"Roll No.: {student["roll_no"]}")
            print(f"Course:   {student["course"]}")
            print(f"Marks:    {student["marks"]}")
            print("----------------")

            del records[index]  # del to delete with speafic key in this case del records[index] means the user get roll no check persent and also check index according to search_student() the index come now suppose index = 1 so deleet  'list of dic 1'
            print("record Deleted Sucessfully...")
    else:
        print("----------------") 
        print("unable to to delete !! Record Not Found!!!")

# 6. Sort Record – Sorting students records by marks (descending order).

def sort():
    if records:
        records.sort(key=lambda x:x["marks"], reverse=True) # so in list list=[name.roll,course,marks] so marks is in 3 index but in dic we assign keys so to sort marks required 
        print("-----------------")
        print("Student sorted Sucessfully....")

    else:
        print("Record Not found!!")    







while True:
    print("================")
    print("Type No. To perform Task")
    print("------------------------")
    print("1 --> Add Students")
    print("2 --> Display Students Data")
    print("3 --> Search Student")
    print("4 --> Update Student Marks")
    print("5 --> Delete Students")
    print("6 --> Sort Students --> By Marks")
    print("7 --> Exit")
    print("================")

    Choice = input("Enter Your choice: ")

    if Choice == "1":
        add_student()
    elif Choice == "2":
        display_students()
    elif Choice == "3":
        student, _ = search_student() # in line 59 the returned saved in 'student' and '_' here _ is used to ignore the index value 
        if student:  # if student is not None
            print("----------------")
            print(f"Name:     {student["name"]}")
            print(f"Roll No.: {student["roll_no"]}")
            print(f"Course:   {student["course"]}")
            print(f"Marks:    {student["marks"]}")
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
