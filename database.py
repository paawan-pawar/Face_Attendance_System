import sqlite3
import os
from datetime import datetime
import threading

class Database:
    def __init__(self):
        # mark_attendance runs in a background thread; allow cross-thread use.
        # We still serialize operations with a lock because the sqlite3 connection
        # object isn't safe for concurrent use.
        self.conn = sqlite3.connect('attendance_system.db', check_same_thread=False)
        self._lock = threading.Lock()
        try:
            self.conn.execute('PRAGMA foreign_keys = ON')
        except Exception:
            pass
        self.create_tables()
    
    def create_tables(self):
        with self._lock:
            cursor = self.conn.cursor()

            # Students table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    enrollment_no TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT,
                    phone TEXT,
                    course TEXT,
                    semester INTEGER,
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Attendance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    enrollment_no TEXT NOT NULL,
                    date DATE NOT NULL,
                    time TIME NOT NULL,
                    status TEXT DEFAULT 'Present',
                    FOREIGN KEY (enrollment_no) REFERENCES students(enrollment_no),
                    UNIQUE(enrollment_no, date)
                )
            ''')

            self.conn.commit()
    
    def register_student(self, enrollment_no, name, email, phone, course, semester):
        try:
            with self._lock:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO students (enrollment_no, name, email, phone, course, semester)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (enrollment_no, name, email, phone, course, semester))
                self.conn.commit()
            return True, "Student registered successfully!"
        except sqlite3.IntegrityError:
            return False, "Enrollment number already exists!"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def mark_attendance(self, enrollment_no):
        try:
            today_date = datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.now().strftime('%H:%M:%S')
            
            with self._lock:
                cursor = self.conn.cursor()
                cursor.execute('''
                    INSERT INTO attendance (enrollment_no, date, time)
                    VALUES (?, ?, ?)
                ''', (enrollment_no, today_date, current_time))
                self.conn.commit()
            return True, "Attendance marked successfully!"
        except sqlite3.IntegrityError:
            return False, "Attendance already marked for today!"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def get_student_info(self, enrollment_no):
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT * FROM students WHERE enrollment_no = ?', (enrollment_no,))
            return cursor.fetchone()
    
    def get_today_attendance(self):
        today_date = datetime.now().strftime('%Y-%m-%d')
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT a.enrollment_no, s.name, a.time 
                FROM attendance a
                JOIN students s ON a.enrollment_no = s.enrollment_no
                WHERE a.date = ?
                ORDER BY a.time
            ''', (today_date,))
            return cursor.fetchall()
    
    def get_all_students(self):
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT enrollment_no, name, course, semester FROM students')
            return cursor.fetchall()
    
    def get_attendance_report(self, enrollment_no=None, start_date=None, end_date=None):
        query = '''
            SELECT a.date, a.time, s.name, a.enrollment_no
            FROM attendance a
            JOIN students s ON a.enrollment_no = s.enrollment_no
            WHERE 1=1
        '''
        params = []
        
        if enrollment_no:
            query += ' AND a.enrollment_no = ?'
            params.append(enrollment_no)
        
        if start_date:
            query += ' AND a.date >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND a.date <= ?'
            params.append(end_date)
        
        query += ' ORDER BY a.date DESC, a.time DESC'

        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def close(self):
        with self._lock:
            self.conn.close()