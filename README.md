Attendance Tracking System with Face Recognition
A comprehensive college attendance tracking system that uses facial recognition technology to automate the attendance marking process. Students can register once and then mark their attendance simply by showing their face to the camera.

📋 Table of Contents
Features

System Requirements

Installation

Usage Guide

System Architecture

File Structure

Troubleshooting

Future Enhancements

License

✨ Features
Student Registration
Register students with detailed information (Name, Enrollment No, Email, Phone, Course, Semester)

Capture and store multiple face samples (20 images) for better recognition accuracy

Automatic face detection using Haar Cascade classifier

Real-time feedback during face capture

Attendance Management
Mark attendance using real-time face recognition

LBPH (Local Binary Patterns Histograms) algorithm for face recognition

Prevents duplicate attendance marking on the same day

Confidence-based recognition with 80% threshold

Visual feedback during recognition process

Reporting & Analytics
View today's attendance in real-time

Filter attendance records by enrollment number and date range

Export attendance reports to CSV format

View complete student list

User Interface
Tab-based intuitive GUI built with Tkinter

Real-time camera feed display

Progress indicators and status messages

Multi-threaded operations for smooth user experience

🖥️ System Requirements
Hardware Requirements
Webcam (built-in or external)

Minimum 4GB RAM (8GB recommended)

100MB free disk space (increases with more student registrations)

Processor: Intel Core i3 or equivalent (i5 recommended for better performance)

Software Requirements
Windows 10/11, Linux, or macOS

Python 3.7 or higher

Webcam drivers properly installed

🚀 Usage Guide
Starting the Application
bash
python main_app.py
1. Registering a New Student
Navigate to the "Register Student" tab

Fill in all student details:

Enrollment Number (unique identifier)

Full Name

Email Address

Phone Number

Course Name

Semester (numeric value)

Click "Register Student & Capture Face"

Position your face clearly in front of the camera

Press SPACE to capture face images (20 images needed)

Press Q to quit early if needed

Wait for model training to complete

Tips for Face Capture:

Ensure good lighting conditions

Look directly at the camera

Vary your head position slightly between captures

Remove glasses if they cause glare

Keep a neutral facial expression

2. Marking Attendance
Navigate to the "Mark Attendance" tab

Click "Mark Attendance (Face Recognition)"

Position your face clearly in front of the camera

Wait for the system to recognize you (shown by green box and name)

Press SPACE to mark attendance

Look for confirmation message

Press Q to exit camera mode

Recognition Confidence:

Green box: Recognized with >80% confidence

Red box: Unknown or low confidence (<80%)

Wait for stable recognition before marking attendance

3. Viewing Attendance Records
Today's Attendance:

Automatically displayed in the "Mark Attendance" tab

Shows all students who marked attendance today with timestamps

Historical Reports:

Navigate to "View Attendance" tab

Apply filters (optional):

Enrollment Number

Start Date (YYYY-MM-DD)

End Date (YYYY-MM-DD)

Click "Apply Filters" to view results

Click "Export to CSV" to download report

4. Managing Students
View all registered students in the "Student List" tab

See enrollment numbers, names, courses, and semesters

Click "Refresh" to update the list

🏗️ System Architecture
Components
text
attendance_system/
│
├── main_app.py           # Main GUI application
├── database.py           # SQLite database operations
├── face_recognizer.py    # Face detection & recognition logic
├── training_data/        # Stored face images (auto-created)
├── attendance_system.db  # SQLite database (auto-created)
├── face_recognizer.yml   # Trained model (auto-created)
└── face_labels.pkl       # Label mappings (auto-created)
Database Schema
Students Table:

id (Primary Key)

enrollment_no (Unique)

name

email

phone

course

semester

registration_date

Attendance Table:

id (Primary Key)

enrollment_no (Foreign Key)

date

time

status

Recognition Algorithm
The system uses LBPH (Local Binary Patterns Histograms) for face recognition:

Converts faces to grayscale

Extracts local binary patterns

Creates histograms for comparison

Uses Euclidean distance for matching

Confidence threshold: 80%

Face Detection
Uses Haar Cascade Classifier:

Pre-trained model from OpenCV

Fast and efficient for real-time detection

Scale factor: 1.1

Minimum neighbors: 5

Minimum face size: 30x30 pixels

🔧 Troubleshooting
Common Issues and Solutions
1. Camera Not Working
bash
Error: Could not open webcam
Solutions:

Check if camera is properly connected

Close other applications using the camera

Check camera permissions in system settings

Try restarting the application

2. Face Not Detected
Solutions:

Ensure adequate lighting

Position face directly in front of camera

Remove obstacles (glasses, masks, hats)

Move closer to camera

3. Low Recognition Confidence
Solutions:

Re-register with better quality images

Capture images in good lighting

Train with more varied head poses

Increase confidence threshold in code (default: 80)

4. Import Errors
bash
ModuleNotFoundError: No module named 'cv2'
Solution: Reinstall OpenCV

bash
pip uninstall opencv-python
pip install opencv-python
5. Database Errors
Solution: Delete database file and restart

bash
del attendance_system.db  # Windows
rm attendance_system.db   # Linux/Mac
6. NumPy Compatibility Issues
Solution: Install compatible versions

bash
pip uninstall numpy pandas -y
pip install numpy==1.24.3 pandas==2.0.3
Performance Optimization Tips
Recognition Speed:

Reduce captured image size (currently 100x100)

Use better lighting for faster detection

Close unnecessary background applications

Storage Management:

Training data folder grows with each registration

Archive old attendance records periodically

Delete training data for deregistered students

Accuracy Improvements:

Capture 20-30 images per student

Use consistent lighting conditions

Train model after each new registration

🚀 Future Enhancements
Potential features to add:

Multiple face detection for batch attendance

Anti-spoofing detection (prevent photo attacks)

Mobile app integration

Email/SMS notifications for absent students

Dashboard with analytics and charts

QR code backup attendance method

Cloud backup and synchronization

Parent/guardian portal

Automated report generation

Voice feedback for visually impaired

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

👥 Contributors
Your Name - Paawan Pawar

🙏 Acknowledgments
OpenCV team for computer vision libraries

Tkinter for GUI framework

SQLite for lightweight database

📞 Support
For issues, questions, or contributions:

Open an issue on GitHub

Contact: [paaw4nnn.2005@gmail.com]

📊 Version History
v1.0.0 (Current)

Initial release

Basic face recognition attendance

Student registration

Attendance reporting

Made by Paawan for educational purpose

