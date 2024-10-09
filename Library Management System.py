'''
LIBRARY MANAGEMENT SYSTEM
A Library Management System (LMS) given access to and manages the resources in your library.
A well-choosen system will increase your library's efficiency, save valuable administration time,
lead to a better educational experience for pupils and help develop independent learning.

PROJECT_WORK.....
Create following menu options:
1. BOOK ISSUE
2. BOOK DEPOSIT
3. ADMINISTRATION MENU
  . CREATE STUDENT RECORD
  . DISPLAY ALL STUDENTS RECORD
  . DISPLAY SPECIFIC STUDENT RECORD
  . MODIFY STUDENT RECORD
  . DELETE STUDENT RECORD
  . CREATE BOOK
  . DISPLAY ALL BOOKS
  . DISPLAY SPECIFIC BOOK
  . MODIFY BOOK
  . DELETE BOOK RECORD
4. EXIT

'''

import pickle


# create 2 files to store the data permanently: books.dat and students.dat to store book info and student info.
books_file = "books.dat"
students_file = "students.dat"

'''

"ab" mode for appending file............ 
"rb" mode for reading file.............. 
"wb" mode for writing file..............
'''

def create_student_record():
    student_id = input('Enter student_id:')
    name = input('Enter student name:')
    with open(students_file, "ab") as file:
        pickle.dump({'id': student_id, 'name': name}, file)
        print("Student record created successfully!")

def display_all_students():
    try:
        with open(students_file, 'rb') as file:
            while True:
                student = pickle.load(file)
                print(student)
    except EOFError:
        pass
    except FileNotFoundError:
        print('No record found.')

def display_specific_student():
    student_id = input("Enter student_id: ")
    found = False
    try:
        with open(students_file, 'rb') as file:
            while True:
                student = pickle.load(file)
                if student['id'] == student_id:
                    print(student)
                    found = True
                    break
    except EOFError:
        if not found:
            print('Student record not found.')
    except FileNotFoundError:
        print('No record found.')

def modify_student_record():
    student_id = input("Enter student_id:")
    found = False
    students = []
    try:
        with open(students_file, 'rb') as file:
            while True:
                student = pickle.load(file)
                if student['id'] == student_id:
                    student['name'] = input("Enter new name:")
                    found = True
                students.append(student)
    except EOFError:
        if found:
            with open(students_file, 'wb') as file:
                for student in students:
                    pickle.dump(student, file)
            print("Student record updated successfully!")
        else:
            print('Student record not found.')
    except FileNotFoundError:
        print("No record found")

def delete_student_record():
    student_id = input("Enter student_id:")
    found = False
    students = []
    try:
        with open(students_file, 'rb') as file:
            while True:
                student = pickle.load(file)
                if student['id'] != student_id:
                    students.append(student)
                else:
                    found = True
    except EOFError:
        if found:
            with open(students_file, 'wb') as file:
                for student in students:
                    pickle.dump(student, file)
            print('Student record deleted successfully!')
        else:
            print("Student record not found.")
    except FileNotFoundError:
        print('No record found.')

def create_book():
    book_id = input('Enter book_id:')
    title = input('Enter title:')
    with open(books_file, "ab") as file:  # "ab" mode for appending binary data
        pickle.dump({'id': book_id, 'title': title}, file)
        print("Book record created successfully!")

def display_all_books():
    try:
        with open(books_file, 'rb') as file:
            while True:
                book = pickle.load(file)
                print(book)
    except EOFError:
        pass
    except FileNotFoundError:
        print('No record found.')

def display_specific_book():
    book_id = input("Enter book_id: ")
    found = False
    try:
        with open(books_file, 'rb') as file:
            while True:
                book = pickle.load(file)
                if book['id'] == book_id:
                    print(book)
                    found = True
                    break
    except EOFError:
        if not found:
            print('Book record not found.')
    except FileNotFoundError:
        print('No record found.')

def modify_book_record():
    book_id = input("Enter book_id:")
    found = False
    books = []
    try:
        with open(books_file, 'rb') as file:
            while True:
                book = pickle.load(file)
                if book['id'] == book_id:
                    book['title'] = input("Enter new title:")
                    found = True
                books.append(book)
    except EOFError:
        if found:
            with open(books_file, 'wb') as file:
                for book in books:
                    pickle.dump(book, file)
            print("Book record updated successfully!")
        else:
            print('Book record not found.')
    except FileNotFoundError:
        print("No record found")

def delete_book_record():
    book_id = input("Enter book_id:")
    found = False
    books = []
    try:
        with open(books_file, 'rb') as file:
            while True:
                book = pickle.load(file)
                if book['id'] != book_id:
                    books.append(book)
                else:
                    found = True
    except EOFError:
        if found:
            with open(books_file, 'wb') as file:
                for book in books:
                    pickle.dump(book, file)
            print('Book record deleted successfully!')
        else:
            print("Book record not found.")
    except FileNotFoundError:
        print('No record found.')

def book_issue():
    student_id = input("Enter student_id:")
    book_id = input("Enter book_id:")
    print(f"Book {book_id} issued to student {student_id}")

def book_deposit():
    student_id = input("Enter student_id:")
    book_id = input("Enter book_id:")
    print(f"Book {book_id} deposited by student {student_id}")

# MAIN MENU
def main_menu():
    while True:
        print("\nMAIN MENU")
        print("1. BOOK ISSUE")
        print("2. BOOK DEPOSIT")
        print("3. ADMINISTRATION MENU")
        print("4. EXIT")
        choice = input("Enter your choice:")

        if choice == '1':
            book_issue()
        elif choice == '2':
            book_deposit()
        elif choice == '3':
            admin_menu()
        elif choice == '4':
            print("Exiting Library Management System. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def admin_menu():
    while True:
        print("\nADMINISTRATION MENU")
        print("1. CREATE STUDENT RECORD")
        print("2. DISPLAY ALL STUDENTS RECORD")
        print("3. DISPLAY SPECIFIC STUDENT RECORD")
        print("4. MODIFY STUDENT RECORD")
        print("5. DELETE STUDENT RECORD")
        print("6. CREATE BOOK")
        print("7. DISPLAY ALL BOOKS")
        print("8. DISPLAY SPECIFIC BOOK")
        print("9. MODIFY BOOK")
        print("10. DELETE BOOK RECORD")
        print("11. BACK TO MAIN MENU")
        choice = input("Enter your choice: ")
        if choice == '1':
            create_student_record()
        elif choice == '2':
            display_all_students()
        elif choice == '3':
            display_specific_student()
        elif choice == '4':
            modify_student_record()
        elif choice == '5':
            delete_student_record()
        elif choice == '6':
            create_book()
        elif choice == '7':
            display_all_books()
        elif choice == '8':
            display_specific_book()
        elif choice == '9':
            modify_book_record()
        elif choice == '10':
            delete_book_record()
        elif choice == '11':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
