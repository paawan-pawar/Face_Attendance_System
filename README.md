# Attendance Tracking System with Face Recognition

A college attendance tracking system that uses face recognition to automate attendance marking. Students register once and then mark attendance by showing their face to the camera.

## Table of contents

- [Features](#features)
- [System requirements](#system-requirements)
- [Installation](#installation)
- [Usage guide](#usage-guide)
- [System architecture](#system-architecture)
- [Project structure & generated files](#project-structure--generated-files)
- [Troubleshooting](#troubleshooting)
- [Future enhancements](#future-enhancements)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contributors](#contributors)
- [Support](#support)
- [Version history](#version-history)

## Features

### Student registration

- Register students with details (Name, Enrollment No, Email, Phone, Course, Semester)
- Capture a face sample from the webcam (press **SPACE** when your face is detected)
- Automatic face detection using a Haar Cascade classifier (OpenCV)
- Real-time feedback during capture

### Attendance management

- Mark attendance using real-time face recognition
- Prevents duplicate attendance marking on the same day (enforced by the database)
- Confidence-based recognition (default threshold: `0.6`)
- Visual feedback during recognition (green = recognized, red = unknown/low confidence)

### Reporting & analytics

- View today's attendance in real-time
- Filter attendance records by enrollment number and date range
- Export attendance reports to CSV
- View complete student list

### User interface

- Tab-based GUI built with Tkinter
- Real-time camera feed windows for registration and attendance
- Multi-threaded attendance marking for smooth UI

## System requirements

### Hardware

- Webcam (built-in or external)
- Minimum 4GB RAM (8GB recommended)
- Disk space: starts small, grows as registrations increase
- Processor: Intel Core i3 or equivalent (i5 recommended for better performance)

### Software

- Windows 10/11, Linux, or macOS
- Python 3.7+

## Installation

1) Create and activate a virtual environment (recommended):

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

2) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional (alternative recognition backend):

- If you install the `face_recognition` package, the app will use it automatically.
- On Windows, installing `face_recognition` can require additional build tools (because it depends on `dlib`). If it’s not installed, the app falls back to an OpenCV LBPH-based recognizer.

## Usage guide

### Start the application

```bash
python main_app.py
```

### 1) Registering a new student

1. Open the **Register Student** tab
2. Fill all details:
	- Enrollment Number (unique)
	- Full Name
	- Email Address
	- Phone Number
	- Course
	- Semester (numeric)
3. Click **Register Student & Capture Face**
4. A camera window opens:
	- Press **SPACE** to capture/register the face
	- Press **Q** to quit without registering

Tips for better capture:

- Ensure good lighting
- Look directly at the camera
- Avoid glare (glasses can sometimes reduce detection quality)

### 2) Marking attendance

1. Open the **Mark Attendance** tab
2. Click **Mark Attendance (Face Recognition)**
3. A camera window opens:
	- Wait until you are recognized (green box + label)
	- Press **SPACE** to mark attendance
	- Press **Q** to exit

Notes:

- The on-screen label shows the recognized student and a confidence score (0–1).
- If attendance for the same enrollment is already marked for today, the database will block duplicates.

### 3) Viewing attendance records

1. Open the **View Attendance** tab
2. (Optional) Set filters:
	- Enrollment Number
	- Start Date (`YYYY-MM-DD`)
	- End Date (`YYYY-MM-DD`)
3. Click **Apply Filters**
4. Click **Export to CSV** to save a report in the project folder

### 4) Viewing the student list

- Open the **Student List** tab
- Click **Refresh** to reload

## System architecture

### Components

- `main_app.py`: Tkinter GUI (tabs for registration, attendance, reports, student list)
- `database.py`: SQLite database operations (students + attendance tables)
- `face_recognizer.py`: Face detection/recognition logic

### Database schema

**Students**

- `enrollment_no` is unique

**Attendance**

- Unique constraint on `(enrollment_no, date)` prevents duplicates for the same day

### Recognition backend

The app supports two recognition backends:

1) **`face_recognition` backend (if installed)**

- Uses face encodings and distance-based matching
- Stores encodings in `face_encodings.pkl`

2) **OpenCV LBPH backend (default fallback)**

- Face detection: Haar cascade from OpenCV
- Recognition: LBPH (`cv2.face.LBPHFaceRecognizer_create`)
- Stores face samples in `face_data/` and label mappings in `face_data/labels.json`

Default recognition threshold: `0.6` (higher = stricter).

## Project structure & generated files

### Project files

```text
attendance_system/
├── main_app.py
├── database.py
├── face_recognizer.py
├── requirements.txt
└── face_data/
	 └── labels.json
```

### Auto-generated files (created at runtime)

- `attendance_system.db`: SQLite database
- `attendance_report_*.csv`: Exported attendance reports
- `face_encodings.pkl`: Only when using the `face_recognition` backend
- `face_data/*.png`: Face samples saved by the OpenCV backend

## Troubleshooting

### Camera not working

Symptoms:

- Camera window is black or the webcam can’t be opened

Fixes:

- Close other apps that may be using the camera
- Check OS camera permissions
- Try reconnecting the webcam / restarting the app

### `ModuleNotFoundError: No module named 'cv2'`

```bash
python -m pip install -r requirements.txt
```

### OpenCV LBPH not available (`cv2.face` missing)

- This requires **opencv-contrib**.

```bash
python -m pip install opencv-contrib-python
```

### Low recognition confidence

- Improve lighting and keep your face centered
- Re-register the student with a clearer capture (if you change the code to capture multiple samples, accuracy improves further)
- Increase/decrease the threshold by changing `confidence_threshold` in `FaceRecognizer.mark_attendance`

### NumPy / Pandas compatibility issues

If you see binary or import errors, reinstall pinned versions:

```bash
python -m pip uninstall numpy pandas -y
python -m pip install numpy==1.24.3 pandas==2.0.3
```

## Future enhancements

- Multiple face detection for batch attendance
- Anti-spoofing detection (prevent photo attacks)
- Dashboard with analytics and charts
- Email/SMS notifications for absent students
- Cloud backup and synchronization

## License

MIT License.

## Acknowledgments

- OpenCV team (computer vision libraries)
- Tkinter (GUI framework)
- SQLite (lightweight embedded database)

## Contributors

- Paawan Pawar

## Support

- Open an issue on GitHub
- Contact: paaw4nnn.2005@gmail.com

## Version history

- v1.0.0 — Initial release (basic registration, face-based attendance, reporting)

