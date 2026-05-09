import cv2 as cv

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(frame):
    # Convert frame to grayscale because Haar Cascade works on intensisty patterns
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # .detectMultiScale scans the image at different scales
    # scaleFactor = 1.1 means the detector slowly resizes the search window by 10 percent
    # minNeighbors = 5 means a face must be confirmed by multiple nearby detections
    # minSize = (60, 60) ignores very tiny face detections
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    return faces

def get_largest_face(faces):
    if len(faces) == 0: 
        return None
    
    # Choose face with biggest area
    largest_face = max(faces, key=lambda face: face[2] * face[3])

    return largest_face

def draw_face_box(frame, face_box, color):

    if face_box is None:
        return
    
    x, y, w, h = face_box

    cv.rectangle(frame, (x,y), (x+w, y+h), color, 2)
    cv.putText(frame, "Winner Face", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def crop_face(frame, face_box, padding):
    if face_box is None:
        return None
    
    x, y, w, h = face_box

    frame_height, frame_width = frame.shape[:2]

    # Add padding but keep crop inside the frame
    x1 = max(0, x-padding)
    y1 = max(0, y-padding)
    x2 = min(frame_width, x + w + padding)
    y2 = min(frame_height, y + h + padding)

    face_crop = frame[y1:y2, x1:x2].copy()

    return face_crop