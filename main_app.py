import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from database import Database
from face_recognizer import FaceRecognizer
from datetime import datetime
import pandas as pd

class AttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("College Attendance System - Face Recognition")
        self.root.geometry("1200x700")
        
        # Initialize components
        self.db = Database()
        self.face_recognizer = FaceRecognizer()
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Register Student
        self.register_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.register_frame, text="Register Student")
        self.setup_register_tab()
        
        # Tab 2: Mark Attendance
        self.attendance_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.attendance_frame, text="Mark Attendance")
        self.setup_attendance_tab()
        
        # Tab 3: View Attendance
        self.view_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.view_frame, text="View Attendance")
        self.setup_view_tab()
        
        # Tab 4: Student List
        self.student_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.student_frame, text="Student List")
        self.setup_student_tab()
    
    def setup_register_tab(self):
        # Create main frame with padding
        main_frame = ttk.Frame(self.register_frame, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Student Registration", 
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Form fields
        fields = [
            ("Enrollment Number:", "enrollment"),
            ("Full Name:", "name"),
            ("Email:", "email"),
            ("Phone Number:", "phone"),
            ("Course:", "course"),
            ("Semester:", "semester")
        ]
        
        self.register_entries = {}
        
        for i, (label, key) in enumerate(fields, start=1):
            ttk.Label(main_frame, text=label, font=('Arial', 10)).grid(
                row=i, column=0, sticky=tk.W, pady=5)
            entry = ttk.Entry(main_frame, width=40, font=('Arial', 10))
            entry.grid(row=i, column=1, pady=5, padx=(10, 0))
            self.register_entries[key] = entry
        
        # Register button
        self.register_btn = ttk.Button(main_frame, text="Register Student & Capture Face", 
                                       command=self.register_student)
        self.register_btn.grid(row=len(fields)+1, column=0, columnspan=2, pady=20)
        
        # Status label
        self.register_status = ttk.Label(main_frame, text="", foreground="blue")
        self.register_status.grid(row=len(fields)+2, column=0, columnspan=2)
    
    def setup_attendance_tab(self):
        main_frame = ttk.Frame(self.attendance_frame, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Mark Attendance", 
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Today's date
        today_date = datetime.now().strftime("%Y-%m-%d")
        date_label = ttk.Label(main_frame, text=f"Date: {today_date}", 
                               font=('Arial', 12))
        date_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Mark attendance button
        self.mark_btn = ttk.Button(main_frame, text="Mark Attendance (Face Recognition)", 
                                   command=self.mark_attendance, width=40)
        self.mark_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Today's attendance list
        ttk.Label(main_frame, text="Today's Attendance:", 
                  font=('Arial', 12, 'bold')).grid(row=3, column=0, columnspan=2, pady=(20, 10))
        
        # Treeview for attendance list
        columns = ('Enrollment', 'Name', 'Time')
        self.attendance_tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.attendance_tree.heading(col, text=col)
            self.attendance_tree.column(col, width=150)
        
        self.attendance_tree.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.attendance_tree.yview)
        scrollbar.grid(row=4, column=2, sticky='ns')
        self.attendance_tree.configure(yscrollcommand=scrollbar.set)
        
        # Refresh button
        refresh_btn = ttk.Button(main_frame, text="Refresh", command=self.refresh_attendance_list)
        refresh_btn.grid(row=5, column=0, columnspan=2, pady=10)
        
        # Initial load
        self.refresh_attendance_list()
    
    def setup_view_tab(self):
        main_frame = ttk.Frame(self.view_frame, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="View Attendance Report", 
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Filters
        ttk.Label(main_frame, text="Enrollment Number:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.filter_enrollment = ttk.Entry(main_frame, width=20)
        self.filter_enrollment.grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(main_frame, text="Start Date:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.filter_start_date = ttk.Entry(main_frame, width=20)
        self.filter_start_date.grid(row=2, column=1, pady=5, padx=5)
        self.filter_start_date.insert(0, "2024-01-01")
        
        ttk.Label(main_frame, text="End Date:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.filter_end_date = ttk.Entry(main_frame, width=20)
        self.filter_end_date.grid(row=3, column=1, pady=5, padx=5)
        self.filter_end_date.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        # Buttons
        filter_btn = ttk.Button(main_frame, text="Apply Filters", command=self.apply_filters)
        filter_btn.grid(row=4, column=0, pady=10)
        
        export_btn = ttk.Button(main_frame, text="Export to CSV", command=self.export_attendance)
        export_btn.grid(row=4, column=1, pady=10)
        
        # Treeview for attendance report
        columns = ('Date', 'Time', 'Name', 'Enrollment')
        self.report_tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.report_tree.heading(col, text=col)
            self.report_tree.column(col, width=150)
        
        self.report_tree.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.report_tree.yview)
        scrollbar.grid(row=5, column=3, sticky='ns')
        self.report_tree.configure(yscrollcommand=scrollbar.set)
        
        # Initial load
        self.apply_filters()
    
    def setup_student_tab(self):
        main_frame = ttk.Frame(self.student_frame, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Student List", 
                                font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Treeview for students
        columns = ('Enrollment', 'Name', 'Course', 'Semester')
        self.student_tree = ttk.Treeview(main_frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.student_tree.heading(col, text=col)
            self.student_tree.column(col, width=200)
        
        self.student_tree.grid(row=1, column=0, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.student_tree.yview)
        scrollbar.grid(row=1, column=1, sticky='ns')
        self.student_tree.configure(yscrollcommand=scrollbar.set)
        
        # Refresh button
        refresh_btn = ttk.Button(main_frame, text="Refresh", command=self.refresh_student_list)
        refresh_btn.grid(row=2, column=0, pady=10)
        
        # Initial load
        self.refresh_student_list()
    
    def register_student(self):
        # Get form data
        enrollment = self.register_entries['enrollment'].get().strip()
        name = self.register_entries['name'].get().strip()
        email = self.register_entries['email'].get().strip()
        phone = self.register_entries['phone'].get().strip()
        course = self.register_entries['course'].get().strip()
        semester = self.register_entries['semester'].get().strip()
        
        # Validate
        if not all([enrollment, name, email, phone, course, semester]):
            messagebox.showerror("Error", "Please fill all fields!")
            return
        
        try:
            semester = int(semester)
        except ValueError:
            messagebox.showerror("Error", "Semester must be a number!")
            return
        
        # Register in database
        success, message = self.db.register_student(enrollment, name, email, phone, course, semester)
        
        if success:
            self.register_status.config(text="Database registration successful. Now capturing face...")
            self.root.update()
            
            # Register face
            face_success = self.face_recognizer.register_face(name, enrollment)
            
            if face_success:
                self.register_status.config(text=f"Student {name} registered successfully!")
                messagebox.showinfo("Success", f"Student {name} registered successfully with face!")
                # Clear form
                for entry in self.register_entries.values():
                    entry.delete(0, tk.END)
                self.refresh_student_list()
            else:
                self.register_status.config(text="Face registration failed! Please try again.")
                messagebox.showerror("Error", "Face registration failed! Please try again.")
        else:
            self.register_status.config(text=message)
            messagebox.showerror("Error", message)
    
    def mark_attendance(self):
        # Run in separate thread to not block GUI
        def mark_thread():
            # Disable button on main thread
            self.root.after(0, lambda: self.mark_btn.config(state='disabled', text='Processing...'))

            success = self.face_recognizer.mark_attendance(self.db)

            def finish_ui():
                self.mark_btn.config(state='normal', text='Mark Attendance (Face Recognition)')
                if success:
                    # Refresh both the "Today's Attendance" list and the report tab
                    self.refresh_attendance_list()
                    self.apply_filters()
                    messagebox.showinfo("Success", "Attendance marked successfully!")
                else:
                    messagebox.showwarning("Warning", "No face recognized or attendance already marked!")

            self.root.after(0, finish_ui)
        
        threading.Thread(target=mark_thread, daemon=True).start()
    
    def refresh_attendance_list(self):
        # Clear current items
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        
        # Get today's attendance
        attendance = self.db.get_today_attendance()
        
        for record in attendance:
            self.attendance_tree.insert('', tk.END, values=record)
    
    def apply_filters(self):
        enrollment = self.filter_enrollment.get().strip() or None
        start_date = self.filter_start_date.get().strip() or None
        end_date = self.filter_end_date.get().strip() or None
        
        # Clear current items
        for item in self.report_tree.get_children():
            self.report_tree.delete(item)
        
        # Get filtered attendance
        attendance = self.db.get_attendance_report(enrollment, start_date, end_date)
        
        for record in attendance:
            self.report_tree.insert('', tk.END, values=record)
    
    def export_attendance(self):
        enrollment = self.filter_enrollment.get().strip() or None
        start_date = self.filter_start_date.get().strip() or None
        end_date = self.filter_end_date.get().strip() or None
        
        attendance = self.db.get_attendance_report(enrollment, start_date, end_date)
        
        if attendance:
            df = pd.DataFrame(attendance, columns=['Date', 'Time', 'Name', 'Enrollment'])
            filename = f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            messagebox.showinfo("Success", f"Report exported to {filename}")
        else:
            messagebox.showwarning("Warning", "No data to export!")
    
    def refresh_student_list(self):
        # Clear current items
        for item in self.student_tree.get_children():
            self.student_tree.delete(item)
        
        # Get all students
        students = self.db.get_all_students()
        
        for student in students:
            self.student_tree.insert('', tk.END, values=student)
    
    def on_closing(self):
        self.db.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()