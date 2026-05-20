from __future__ import annotations

from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from PIL import ImageDraw

try:
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer  # type: ignore
except ModuleNotFoundError:
    VideoProcessorBase = object  # type: ignore
    WebRtcMode = None  # type: ignore
    webrtc_streamer = None  # type: ignore

from database import Database
from face_recognizer import FaceRecognizer


def _count_opencv_samples() -> int:
    import os

    face_dir = "face_data"
    if not os.path.isdir(face_dir):
        return 0
    count = 0
    for name in os.listdir(face_dir):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            count += 1
    return count


def _get_db() -> Database:
    if "db" not in st.session_state:
        st.session_state.db = Database()
    return st.session_state.db


def _get_recognizer() -> FaceRecognizer:
    if "recognizer" not in st.session_state:
        st.session_state.recognizer = FaceRecognizer()
    return st.session_state.recognizer


def _file_to_rgb_array(file) -> np.ndarray:
    """Read Streamlit UploadedFile/camera input robustly into RGB numpy array."""
    if hasattr(file, "getvalue"):
        data = file.getvalue()
    else:
        data = file.read()
    image = Image.open(BytesIO(data)).convert("RGB")
    return np.array(image)


def _draw_boxes(rgb: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> Image.Image:
    img = Image.fromarray(rgb)
    if not boxes:
        return img
    draw = ImageDraw.Draw(img)
    for (top, right, bottom, left) in boxes:
        draw.rectangle([(left, top), (right, bottom)], outline=(0, 255, 0), width=3)
    return img


def _pick_largest_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    return max(boxes, key=lambda b: max((b[2] - b[0]), 0) * max((b[1] - b[3]), 0))


def _crop_face(rgb: np.ndarray, box: tuple[int, int, int, int], pad: float = 0.20) -> Image.Image:
    top, right, bottom, left = box
    h, w = rgb.shape[:2]
    pad_x = int((right - left) * pad)
    pad_y = int((bottom - top) * pad)
    x0 = max(left - pad_x, 0)
    y0 = max(top - pad_y, 0)
    x1 = min(right + pad_x, w)
    y1 = min(bottom + pad_y, h)
    return Image.fromarray(rgb[y0:y1, x0:x1])


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


class _FaceBoxProcessor(VideoProcessorBase):
    def __init__(self, recognizer: FaceRecognizer):
        self._recognizer = recognizer
        self.last_rgb: np.ndarray | None = None
        self.last_boxes: list[tuple[int, int, int, int]] = []

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = img_bgr[:, :, ::-1]
        self.last_rgb = img_rgb

        boxes = self._recognizer.detect_faces_from_image(img_rgb)
        self.last_boxes = boxes

        # Draw on BGR image for the live preview
        try:
            import cv2  # type: ignore

            for (top, right, bottom, left) in boxes:
                cv2.rectangle(img_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        except Exception:
            pass

        import av  # type: ignore

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


def _capture_widget(prefix: str, recognizer: FaceRecognizer) -> tuple[np.ndarray | None, list[tuple[int, int, int, int]]]:
    """Capture an image via Snapshot or Live (WebRTC). Returns (rgb, boxes)."""

    capture_mode = st.radio(
        "Capture mode",
        options=["Snapshot", "Live (recommended)"],
        horizontal=True,
        key=f"{prefix}_mode",
    )

    rgb: np.ndarray | None = st.session_state.get(f"{prefix}_rgb")
    boxes: list[tuple[int, int, int, int]] = st.session_state.get(f"{prefix}_boxes", [])

    if capture_mode == "Live (recommended)":
        if webrtc_streamer is None:
            st.info("Install `streamlit-webrtc` to use Live capture.")
        else:
            ctx = webrtc_streamer(
                key=f"{prefix}_webrtc",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=lambda: _FaceBoxProcessor(recognizer),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            col_a, col_b = st.columns([1, 1])
            with col_a:
                if st.button("Capture frame", key=f"{prefix}_capture", type="primary"):
                    if ctx.video_processor is None or ctx.video_processor.last_rgb is None:
                        st.warning("Waiting for camera frames…")
                    else:
                        rgb = ctx.video_processor.last_rgb
                        boxes = ctx.video_processor.last_boxes
                        st.session_state[f"{prefix}_rgb"] = rgb
                        st.session_state[f"{prefix}_boxes"] = boxes
                        st.rerun()
            with col_b:
                if rgb is not None:
                    st.caption("Captured frame saved.")

    else:
        camera_img = st.camera_input("Take a photo", key=f"{prefix}_camera")
        upload_img = st.file_uploader(
            "…or upload an image",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            key=f"{prefix}_upload",
        )
        file = camera_img if camera_img is not None else upload_img
        if file is not None:
            rgb = _file_to_rgb_array(file)
            boxes = recognizer.detect_faces_from_image(rgb)
            st.session_state[f"{prefix}_rgb"] = rgb
            st.session_state[f"{prefix}_boxes"] = boxes

    return rgb, boxes


st.set_page_config(
    page_title="Attendance System",
    page_icon="✅",
    layout="wide",
)

st.title("Attendance Tracking System")

_db = _get_db()
_recognizer = _get_recognizer()

_default_threshold = 0.45 if _recognizer.backend == "opencv_lbph" else 0.60

with st.sidebar:
    st.subheader("Status")
    st.write(f"Recognition backend: `{_recognizer.backend}`")
    st.caption(
        "Streamlit uses snapshots (camera photo/upload). "
        "It does not open OpenCV webcam windows on the server."
    )
    if _recognizer.backend == "unavailable":
        st.error(
            "Face recognition backend is unavailable in this Python environment. "
            "Install dependencies from requirements.txt in the same environment you run Streamlit."
        )
    else:
        if _recognizer.backend == "opencv_lbph":
            st.caption(f"Face samples saved: {_count_opencv_samples()}")

register_tab, attendance_tab, report_tab, students_tab = st.tabs(
    ["Register Student", "Mark Attendance", "View Attendance", "Student List"]
)

with register_tab:
    st.subheader("Register Student")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        enrollment_no = st.text_input("Enrollment Number", placeholder="e.g. 22CS101")
        full_name = st.text_input("Full Name", placeholder="e.g. Aditi Sharma")
        email = st.text_input("Email", placeholder="e.g. aditi@example.com")
        phone = st.text_input("Phone Number", placeholder="e.g. 9876543210")
        course = st.text_input("Course", placeholder="e.g. B.Tech CSE")
        semester = st.number_input("Semester", min_value=1, max_value=16, step=1, value=1)

    with col2:
        st.markdown("**Capture Face Photo**")
        rgb, boxes = _capture_widget("register", _recognizer)

        confidence_threshold = st.slider(
            "Recognition strictness",
            min_value=0.30,
            max_value=0.95,
            value=_default_threshold,
            step=0.05,
            help="Used when verifying face detection/encoding quality.",
        )

        if rgb is not None:
            st.caption(f"Faces detected: {len(boxes)}")
            st.image(_draw_boxes(rgb, boxes), use_container_width=True)
            largest = _pick_largest_box(boxes)
            if largest is not None:
                st.caption("Largest face crop")
                st.image(_crop_face(rgb, largest), use_container_width=True)

    action_col1, action_col2 = st.columns([1, 1])

    if action_col1.button("Register Student", type="primary"):
        enrollment_no = enrollment_no.strip()
        full_name = full_name.strip()
        email = email.strip()
        phone = phone.strip()
        course = course.strip()

        if not all([enrollment_no, full_name, email, phone, course]):
            st.error("Please fill all fields.")
        elif rgb is None:
            st.error("Please capture or upload a face photo.")
        else:
            ok, msg = _db.register_student(
                enrollment_no=enrollment_no,
                name=full_name,
                email=email,
                phone=phone,
                course=course,
                semester=int(semester),
            )

            if not ok:
                st.error(msg)
            else:
                face_ok = _recognizer.register_face_from_image(full_name, enrollment_no, rgb)
                if face_ok:
                    st.success("Student registered and face captured successfully!")
                else:
                    st.warning(
                        "Student saved to database, but face registration failed. "
                        "Try again with better lighting / clearer frontal face."
                    )

    if action_col2.button("Enroll/Update Face Only"):
        enrollment_no = enrollment_no.strip()
        if not enrollment_no:
            st.error("Enter an Enrollment Number first.")
        elif rgb is None:
            st.error("Capture a face photo first.")
        else:
            info = _db.get_student_info(enrollment_no)
            if not info:
                st.error("Student not found in DB. Register student first.")
            else:
                # students table: (id, enrollment_no, name, email, phone, course, semester, registration_date)
                stored_name = str(info[2])
                face_ok = _recognizer.register_face_from_image(stored_name, enrollment_no, rgb)
                if face_ok:
                    st.success("Face sample saved/updated for this student.")
                else:
                    st.warning("Face enrollment failed. Try a clearer photo.")

with attendance_tab:
    st.subheader("Mark Attendance")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("**Capture Attendance Photo**")
        rgb, boxes = _capture_widget("attendance", _recognizer)

        threshold = st.slider(
            "Confidence threshold",
            min_value=0.30,
            max_value=0.95,
            value=_default_threshold,
            step=0.05,
        )

        if rgb is not None:
            st.caption(f"Faces detected: {len(boxes)}")
            st.image(_draw_boxes(rgb, boxes), use_container_width=True)
            largest = _pick_largest_box(boxes)
            if largest is not None:
                st.caption("Largest face crop")
                st.image(_crop_face(rgb, largest), use_container_width=True)
            if len(boxes) == 0:
                st.session_state.last_recognition = None
                st.error("No face detected in this capture.")
            else:
                result = _recognizer.recognize_from_image(rgb, confidence_threshold=threshold)
                st.session_state.last_recognition = result

                if result is None:
                    st.warning(
                        "Face detected, but not recognized. "
                        "Try lowering the threshold, and enroll 3–5 samples using 'Enroll/Update Face Only'."
                    )
                else:
                    name, enrollment, conf = result
                    st.success(f"Recognized: {name} ({enrollment}) — confidence {conf:.2f}")

        mark = st.button("Mark Attendance", type="primary")
        if mark:
            result = st.session_state.get("last_recognition")
            if result is None:
                st.error("Capture a photo and get a recognition result first.")
            else:
                _name, enrollment, _conf = result
                ok, msg = _db.mark_attendance(enrollment)
                if ok:
                    st.success(msg)
                else:
                    st.warning(msg)

    with col2:
        st.markdown("**Today's Attendance**")
        today = _db.get_today_attendance()
        if today:
            df_today = pd.DataFrame(today, columns=["Enrollment", "Name", "Time"])
            st.dataframe(df_today, use_container_width=True, hide_index=True)
        else:
            st.info("No attendance marked today yet.")

with report_tab:
    st.subheader("Attendance Report")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        filter_enrollment = st.text_input("Enrollment (optional)", key="filter_enrollment")
    with col2:
        start_date = st.date_input("Start date", value=date(2024, 1, 1))
    with col3:
        end_date = st.date_input("End date", value=date.today())

    enrollment = filter_enrollment.strip() or None
    attendance = _db.get_attendance_report(
        enrollment_no=enrollment,
        start_date=start_date.isoformat() if start_date else None,
        end_date=end_date.isoformat() if end_date else None,
    )

    if attendance:
        df = pd.DataFrame(attendance, columns=["Date", "Time", "Name", "Enrollment"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            data=_df_to_csv_bytes(df),
            file_name="attendance_report.csv",
            mime="text/csv",
        )
    else:
        st.info("No records found for the selected filters.")

with students_tab:
    st.subheader("Students")

    students = _db.get_all_students()
    if students:
        df_students = pd.DataFrame(students, columns=["Enrollment", "Name", "Course", "Semester"])
        st.dataframe(df_students, use_container_width=True, hide_index=True)
    else:
        st.info("No students registered yet.")
