import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import json

try:
    import face_recognition  # type: ignore
except ModuleNotFoundError as e:
    face_recognition = None
    _FACE_RECOGNITION_IMPORT_ERROR = e

class FaceRecognizer:
    def __init__(self):
        self.backend = "face_recognition" if face_recognition is not None else "opencv_lbph"

        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_enrollments = []

        self.encodings_file = 'face_encodings.pkl'

        # OpenCV fallback storage
        self.face_data_dir = 'face_data'
        self.face_labels_file = os.path.join(self.face_data_dir, 'labels.json')
        self._label_to_student: dict[int, tuple[str, str]] = {}
        self._student_to_label: dict[str, int] = {}
        self._lbph_model = None
        self._haar_cascade = None

        if self.backend == "face_recognition":
            self.load_encodings()
        else:
            self._ensure_opencv_backend_ready()
            self._load_opencv_labels()
            self._train_opencv_model()

    def _ensure_face_recognition_available(self) -> bool:
        if face_recognition is not None:
            return True

        print(
            "Face recognition features are unavailable because the 'face_recognition' package "
            "is not installed (or failed to install).\n"
            "Install dependencies with: python -m pip install -r requirements.txt\n"
            "On Windows, you may also need Visual C++ Build Tools for 'dlib'.\n"
            f"Original error: {_FACE_RECOGNITION_IMPORT_ERROR}"
        )
        return False

    def _ensure_opencv_backend_ready(self) -> bool:
        os.makedirs(self.face_data_dir, exist_ok=True)

        # Haar cascade for face detection (ships with OpenCV)
        if self._haar_cascade is None:
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)
            if self._haar_cascade.empty():
                print(f"Failed to load Haar cascade at: {cascade_path}")
                return False

        # LBPH face recognizer requires opencv-contrib-python (cv2.face)
        if not hasattr(cv2, 'face') or not hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
            print(
                "OpenCV face recognizer (LBPH) is unavailable.\n"
                "Install opencv-contrib-python: python -m pip install opencv-contrib-python\n"
                "Then restart the app."
            )
            return False

        if self._lbph_model is None:
            self._lbph_model = cv2.face.LBPHFaceRecognizer_create()

        return True

    def _load_opencv_labels(self) -> None:
        self._label_to_student = {}
        self._student_to_label = {}

        if not os.path.exists(self.face_labels_file):
            return

        try:
            with open(self.face_labels_file, 'r', encoding='utf-8') as f:
                payload = json.load(f)

            # Stored as strings in JSON; convert to int keys
            raw = payload.get('label_to_student', {})
            for label_str, info in raw.items():
                label = int(label_str)
                name = str(info.get('name', '')).strip()
                enrollment = str(info.get('enrollment', '')).strip()
                if name and enrollment:
                    self._label_to_student[label] = (name, enrollment)
                    self._student_to_label[enrollment] = label
        except Exception as e:
            print(f"Error loading OpenCV labels: {e}")

    def _save_opencv_labels(self) -> None:
        payload = {
            'label_to_student': {
                str(label): {'name': name, 'enrollment': enrollment}
                for label, (name, enrollment) in self._label_to_student.items()
            }
        }
        with open(self.face_labels_file, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

    def _iter_opencv_training_images(self):
        images = []
        labels = []

        for label, (_name, enrollment) in self._label_to_student.items():
            pattern_prefix = f"{enrollment}_"
            for filename in os.listdir(self.face_data_dir):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                if not filename.startswith(pattern_prefix):
                    continue

                path = os.path.join(self.face_data_dir, filename)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(img)
                labels.append(label)

        return images, np.array(labels, dtype=np.int32)

    def _train_opencv_model(self) -> bool:
        if not self._ensure_opencv_backend_ready():
            return False

        images, labels = self._iter_opencv_training_images()
        if len(images) == 0:
            return False

        try:
            self._lbph_model.train(images, labels)
            return True
        except Exception as e:
            print(f"Error training OpenCV LBPH model: {e}")
            return False

    def _detect_faces_opencv(self, gray_frame):
        if not self._ensure_opencv_backend_ready():
            return []

        # returns list of (x, y, w, h)
        faces = self._haar_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )
        return faces

    def _preprocess_face_roi(self, gray_frame, x, y, w, h):
        roi = gray_frame[y:y + h, x:x + w]
        if roi.size == 0:
            return None
        roi = cv2.resize(roi, (200, 200))
        return roi
    
    def load_encodings(self):
        """Load existing face encodings from file"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                    self.known_face_enrollments = data['enrollments']
                print(f"Loaded {len(self.known_face_encodings)} face encodings")
            except Exception as e:
                print(f"Error loading encodings: {e}")
                self.known_face_encodings = []
                self.known_face_names = []
                self.known_face_enrollments = []
    
    def save_encodings(self):
        """Save face encodings to file"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names,
            'enrollments': self.known_face_enrollments
        }
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
    
    def register_face(self, name, enrollment_no):
        """Register a new face by capturing from webcam"""
        if self.backend == "face_recognition":
            if not self._ensure_face_recognition_available():
                return False
        else:
            if not self._ensure_opencv_backend_ready():
                return False

        print(f"Registering face for {name} ({enrollment_no})")
        
        # Open webcam
        video_capture = cv2.VideoCapture(0)
        print("Press 'Space' to capture face, 'q' to quit")
        
        face_encodings = []
        opencv_face_roi = None
        opencv_face_box = None
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.backend == "face_recognition":
                # Find face locations
                face_locations = face_recognition.face_locations(rgb_frame)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._detect_faces_opencv(gray)
                face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
            
            # Draw rectangles around faces
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Get face encoding
                if len(face_locations) > 0 and self.backend == "face_recognition":
                    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)
                    if face_encoding:
                        face_encodings = face_encoding

                if len(face_locations) > 0 and self.backend != "face_recognition":
                    # Use the first detected face
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    x, y, w, h = left, top, right - left, bottom - top
                    roi = self._preprocess_face_roi(gray, x, y, w, h)
                    if roi is not None:
                        opencv_face_roi = roi
                        opencv_face_box = (top, right, bottom, left)
            
            # Show instructions
            cv2.putText(frame, "Press SPACE to capture face", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press Q to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if face_encodings or opencv_face_roi is not None:
                cv2.putText(frame, "Face detected!", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Register Face', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space key
                if self.backend == "face_recognition":
                    if face_encodings:
                        # Register the face
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(name)
                        self.known_face_enrollments.append(enrollment_no)
                        self.save_encodings()
                        print(f"Face registered successfully for {name}")
                        break
                    else:
                        print("No face detected! Please position your face in front of camera.")
                else:
                    if opencv_face_roi is not None and opencv_face_box is not None:
                        # Assign label for this enrollment if new
                        if enrollment_no in self._student_to_label:
                            label = self._student_to_label[enrollment_no]
                        else:
                            label = (max(self._label_to_student.keys()) + 1) if self._label_to_student else 0
                            self._label_to_student[label] = (name, enrollment_no)
                            self._student_to_label[enrollment_no] = label
                            self._save_opencv_labels()

                        # Save cropped face image
                        safe_enrollment = ''.join(c for c in str(enrollment_no) if c.isalnum() or c in ('-', '_'))
                        safe_name = ''.join(c for c in str(name) if c.isalnum() or c in ('-', '_'))
                        filename = f"{safe_enrollment}_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        path = os.path.join(self.face_data_dir, filename)
                        cv2.imwrite(path, opencv_face_roi)

                        # Retrain model after adding new image
                        self._train_opencv_model()
                        print(f"Face registered successfully for {name}")
                        break
                    else:
                        print("No face detected! Please position your face in front of camera.")
            
            elif key == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        if self.backend == "face_recognition":
            return len(face_encodings) > 0
        return opencv_face_roi is not None
    
    def mark_attendance(self, database, confidence_threshold=0.6):
        """Mark attendance by recognizing face"""
        if self.backend == "face_recognition":
            if not self._ensure_face_recognition_available():
                return False
        else:
            if not self._ensure_opencv_backend_ready():
                return False

            # Try to train (in case new registrations happened)
            self._load_opencv_labels()
            self._train_opencv_model()

        video_capture = cv2.VideoCapture(0)
        print("Press 'q' to quit, 'Space' to mark attendance")
        
        recognized_student = None
        recognition_confidence = 0
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            face_names = []
            face_confidence = []

            if self.backend == "face_recognition":
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find face locations and encodings
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for face_encoding in face_encodings:
                    # Compare with known faces
                    if len(self.known_face_encodings) > 0:
                        distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(distances)
                        confidence = 1 - distances[best_match_index]

                        if confidence >= confidence_threshold:
                            name = self.known_face_names[best_match_index]
                            enrollment = self.known_face_enrollments[best_match_index]
                            face_names.append(f"{name} ({enrollment})")
                            face_confidence.append(confidence)

                            if recognized_student is None:
                                recognized_student = (name, enrollment)
                                recognition_confidence = confidence
                        else:
                            face_names.append("Unknown")
                            face_confidence.append(0)
                    else:
                        face_names.append("No registered faces")
                        face_confidence.append(0)
            else:
                # OpenCV LBPH backend
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self._detect_faces_opencv(gray)

                # Keep same drawing pipeline: use (top,right,bottom,left)
                face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]

                for (x, y, w, h) in faces:
                    roi = self._preprocess_face_roi(gray, x, y, w, h)
                    if roi is None or self._lbph_model is None:
                        face_names.append("Unknown")
                        face_confidence.append(0)
                        continue

                    # LBPH returns (label, distance). Lower distance = better.
                    try:
                        pred_label, distance = self._lbph_model.predict(roi)
                    except Exception:
                        pred_label, distance = -1, 9999

                    # Convert to a confidence-like score in [0,1].
                    # LBPH: lower distance = better match. This exponential mapping makes the
                    # existing default threshold (0.6) usable.
                    confidence = float(np.exp(-max(distance, 0.0) / 80.0))

                    if pred_label in self._label_to_student and confidence >= confidence_threshold:
                        name, enrollment = self._label_to_student[pred_label]
                        face_names.append(f"{name} ({enrollment})")
                        face_confidence.append(confidence)

                        if recognized_student is None:
                            recognized_student = (name, enrollment)
                            recognition_confidence = confidence
                    else:
                        face_names.append("Unknown")
                        face_confidence.append(0)
            
            # Draw results
            for (top, right, bottom, left), name, conf in zip(face_locations, face_names, face_confidence):
                if self.backend == "face_recognition":
                    # Scale back up
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                
                # Draw rectangle
                color = (0, 255, 0) if conf >= confidence_threshold else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Draw label
                label = f"{name} ({conf:.2f})" if conf > 0 else name
                cv2.putText(frame, label, (left, top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show instructions
            cv2.putText(frame, "Press SPACE to mark attendance", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press Q to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Mark Attendance', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space key
                if recognized_student:
                    name, enrollment = recognized_student
                    success, message = database.mark_attendance(enrollment)
                    print(f"Attendance for {name}: {message}")
                    
                    # Show confirmation
                    confirmation_frame = frame.copy()
                    cv2.putText(confirmation_frame, message, (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Mark Attendance', confirmation_frame)
                    cv2.waitKey(2000)
                    break
                else:
                    print("No recognized face detected!")
            
            elif key == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        return recognized_student is not None