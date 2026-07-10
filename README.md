# Driver Monitoring System using MediaPipe

## Overview

This project is a real-time **Driver Monitoring System (DMS)** built using **Python**, **OpenCV**, and **MediaPipe Face Landmarker**. It monitors the driver's facial landmarks through a webcam and detects signs of fatigue and distraction to improve road safety.

The system performs multiple safety checks, including eye closure, drowsiness, yawning, gaze direction, head pose, and phone distraction detection. An alarm is triggered whenever prolonged eye closure or microsleep is detected.

---

## Features

* 👁️ Eye Aspect Ratio (EAR) based eye closure detection
* 😴 Drowsiness detection
* ⏱️ Microsleep detection
* 😮 Yawning detection using Mouth Aspect Ratio (MAR)
* 👀 Gaze direction detection
* 📱 Phone distraction detection
* 🙇 Head-down detection
* 📊 PERCLOS (Percentage of Eye Closure) calculation
* 🔔 Audible alarm using Windows Beep
* 📷 Real-time webcam monitoring

---

## Technologies Used

* Python 3.x
* OpenCV
* MediaPipe Tasks API
* NumPy
* Winsound (Windows)

---

## Project Structure

```
Driver-Monitoring-System/
│
├── models/
│   └── face_landmarker.task
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Driver-Monitoring-System.git
cd Driver-Monitoring-System
```

### 2. Create a virtual environment (Optional)

```bash
python -m venv env
```

Activate it:

**Windows**

```bash
env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install opencv-python mediapipe numpy
```

---

## Download the Face Landmarker Model

Download the MediaPipe Face Landmarker model (`face_landmarker.task`) from the official MediaPipe website and place it inside the `models` folder.

Example:

```
models/
    face_landmarker.task
```

---

## Run the Project

```bash
python main.py
```

Press **ESC** to exit the application.

---

## Detection Modules

### Eye Aspect Ratio (EAR)

Calculates the eye opening using facial landmarks. If the EAR falls below a threshold, the system identifies the eyes as closed.

---

### Drowsiness Detection

Measures how long the driver's eyes remain closed. If the duration exceeds the predefined threshold, the system marks the driver as drowsy and activates an alarm.

---

### Microsleep Detection

Detects prolonged eye closure that may indicate microsleep episodes and immediately triggers an alert.

---

### Mouth Aspect Ratio (MAR)

Measures mouth opening to detect yawning.

---

### Gaze Detection

Tracks iris positions to estimate whether the driver is looking:

* Left
* Right
* Forward

---

### Head Pose Detection

Uses nose and chin landmarks to estimate whether the driver's head is tilted downward.

---

### Phone Distraction Detection

Detects possible phone usage when the driver's head is down while the gaze remains approximately centered.

---

### PERCLOS

Calculates the percentage of frames in which the driver's eyes are closed, providing an additional fatigue metric.

---

## Future Improvements

* Face recognition for driver authentication
* Multiple driver support
* Mobile phone detection using YOLO
* Head pose estimation with 3D landmarks
* Emotion recognition
* Driver identity logging
* Fatigue analytics dashboard
* SMS or email alerts
* Cloud-based monitoring
* Performance optimization for embedded devices

---

## Requirements

* Python 3.10 or above
* Webcam
* Windows OS (for `winsound` alarm)

---

## Acknowledgements

* Google MediaPipe
* OpenCV
* NumPy

---

## License

This project is intended for educational and research purposes.

If you upload this project to GitHub, you should also include a `requirements.txt` file. Generate a requirements.txt file for the project.
