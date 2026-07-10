import mediapipe as mp
import cv2
import numpy as np
import winsound
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configuration
MODEL_PATH = r'models\face_landmarker.task'

# Eye Thresholds
EAR_THRESHOLD = 0.20
BLINK_THRESHOLD = 0.70

# Yawing Thresholds
MAR_THRESHOLD = 0.60

# Time Thresholds
DROWSY_TIME = 1.5
MICROSLEEP_TIME = 3.0

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ALARM FUNCTIONS
alarm_playing = False

def play_alarm():
    global alarm_playing

    if not alarm_playing:
        winsound.Beep(2500, 1000)
        alarm_playing = True

def stop_alarm():
    global alarm_playing
    alarm_playing = False

# Mediapipe Initialization

base_options = python.BaseOptions(
    model_asset_path = MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options = base_options,
    output_face_blendshapes = True,
    output_facial_transformation_matrixes = True,
    num_faces = 1
)

detector = vision.FaceLandmarker.create_from_options(options)

# Landmark Indexes

# Eyes
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Iris
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

# Mouth
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308

# Face
NOSE = 1
CHIN = 152


# Helper Functions

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(eye):
    A = distance(eye[1], eye[5])
    B = distance(eye[2], eye[4])
    C = distance(eye[0], eye[3])

    return (A+B)/(2*C)

def compute_mar(landmarks):
    upper = np.array([
        landmarks[UPPER_LIP].x,
        landmarks[UPPER_LIP].y
    ])

    lower = np.array([
        landmarks[LOWER_LIP].x,
        landmarks[LOWER_LIP].y
    ])

    left = np.array([
        landmarks[LEFT_MOUTH].x,
        landmarks[LEFT_MOUTH].y
    ])

    right = np.array([
        landmarks[RIGHT_MOUTH].x,
        landmarks[RIGHT_MOUTH].y
    ])

    vertical = np.linalg.norm(upper - lower)
    horizontal = np.linalg.norm(left - right)

    mar = vertical / horizontal
    return mar

def iris_center(landmarks, indices):
    pts = []
    for idx in indices:
        pts.append([
            landmarks[idx].x,
            landmarks[idx].y
        ])

    pts = np.array(pts)

    return np.mean(pts, axis = 0)

# State Variables

closed_start = None

closed_frames = 0
total_frames = 0

state = 'NORMAL'

start_time = time.time()

# Camera Initilization

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Cannot open camera')
    exit()

# Main Loop
while True:
    success, frame = cap.read()

    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format = mp.ImageFormat.SRGB,
        data = rgb
    )

    results = detector.detect(mp_image)

    total_frames += 1

    if results.face_landmarks:
        landmarks = results.face_landmarks[0]

        # Extract left eye

        left_eye = []
        for idx in LEFT_EYE:
            left_eye.append(
                (
                    landmarks[idx].x,
                    landmarks[idx].y
                )
            )

        # Extract right eye

        right_eye = []
        for idx in RIGHT_EYE:
            right_eye.append(
                (
                    landmarks[idx].x,
                    landmarks[idx].y
                )
            )

        # Ear calculation

        left_ear = compute_ear(left_eye)
        right_ear = compute_ear(right_eye)

        avg_ear = (left_ear + right_ear)/2

        # Blendshape Values

        left_blink = 0
        right_blink = 0

        if results.face_blendshapes:
            blendshapes = results.face_blendshapes[0]

            for item in blendshapes:
                if item.category_name == 'eyeBlinkLeft':
                    left_blink = item.score

                elif item.category_name == 'eyeBlinkRight':
                    right_blink = item.score

        # Eye closed Detection
        eye_closed = False

        if avg_ear < EAR_THRESHOLD or (
            left_blink > BLINK_THRESHOLD and
            right_blink > BLINK_THRESHOLD
        ):
            
            eye_closed = True
            closed_frames += 1

        # Drowsiness timer

        if eye_closed:
            if closed_start is None:
                closed_start = time.time()

            duration = time.time() - closed_start

        else:
            duration = 0
            closed_start = None
            stop_alarm()

        # Driver state

        if duration > MICROSLEEP_TIME:
            state = 'MICROSLEEP'
            play_alarm()

        elif duration > DROWSY_TIME:
            state = 'DROWSY'
            play_alarm()

        else:
            state = 'NORMAL'


        # Yawning Detection

        mar = compute_mar(landmarks)

        yawn_text = ""

        if mar > MAR_THRESHOLD:
            yawn_text = 'YAWNING DETECTED'

        # Gaze Detection

        left_iris = iris_center(landmarks, LEFT_IRIS)
        right_iris = iris_center(landmarks, RIGHT_IRIS)

        gaze_x = (left_iris[0] + right_iris[0])/2

        distraction = 'FORWARD'

        if gaze_x < 0.42:
            distraction = 'LOOKING LEFT'

        elif gaze_x > 0.58:
            distraction = 'LOOKING RIGHT'


        # Head Down Detection

        nose_y = landmarks[NOSE].y
        chin_y = landmarks[CHIN].y

        head_distance = chin_y - nose_y

        head_down = False

        if head_distance < 0.18:
            head_down = True

        # Phone Distraction

        phone_distraction = False
        if head_down and (0.45 < gaze_x < 0.55):
            phone_distraction = True

        # PERCLOS

        perclos = (
            closed_frames /
            max(total_frames, 1)
        ) * 100

        # Display Values

        cv2.putText(
            frame,
            f"EAR : {avg_ear:.2f}",
            (20, 40),
            FONT,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"MAR : {mar:.2f}",
            (20, 80),
            FONT,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"PERCLOS : {perclos:.2f}",
            (20, 120),
            FONT,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Closed Time : {duration:.2f}",
            (20, 160),
            FONT,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            distraction,
            (20, 200),
            FONT,
            0.7,
            (255, 255, 255),
            2
        )

        # Warnings

        if yawn_text != "":
            cv2.putText(
            frame,
            yawn_text,
            (20, 40),
            FONT,
            0.8,
            (0, 0, 255),
            2
        )

        if head_down:
            cv2.putText(
            frame,
            "HEAD DOWN",
            (20, 280),
            FONT,
            0.8,
            (0, 0, 255),
            2
        ) 
            
        if phone_distraction:
            cv2.putText(
            frame,
            "PHONE DISTRACTION",
            (20, 320),
            FONT,
            0.8,
            (0, 0, 255),
            2
        )
            
    # Driver state color

    if state == "NORMAL":
        color = (0, 255, 0)

    elif state == "DOWSY":
        color = (0, 255, 255)

    else:
        color = (0, 0, 255)

    cv2.putText(
        frame,
        state,
        (20, 370),
        FONT,
        1,
        color,
        3
    )

    # Show window

    cv2.imshow(
        "Driver Monitoring System",
        frame
    )
            
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
stop_alarm()


