import cv2
import pytesseract
import tempfile

def extract_ocr_from_video(video_path, frame_rate=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)

    frames_with_ocr = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            if text.strip():
                timestamp = frame_idx / fps
                frames_with_ocr.append({
                    "start": timestamp,
                    "end": timestamp + 2,
                    "text": text.strip()
                })
        frame_idx += 1

    cap.release()
    return frames_with_ocr
