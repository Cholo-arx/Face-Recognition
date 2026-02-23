import cv2
import numpy as np
import os

# --- Configuration ---
dataset_path = "./face_dataset/"
os.makedirs(dataset_path, exist_ok=True)

# Ask for the person's name
name = input("Enter your name: ").strip()
if not name:
    print("Error: Name cannot be empty.")
    exit()

# --- Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
if face_cascade.empty():
    print("Error: Could not load haarcascade_frontalface_alt.xml. Make sure it's in the same folder.")
    cap.release()
    exit()

face_data = []
max_samples = 50

print(
    f"Collecting face data for '{name}'. Look at the camera. Press 'q' to quit early.")

while True:
    ret, frame = cap.read()
    if ret == False:
        print("Warning: Failed to grab frame. Camera may have disconnected.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for face in faces:
        x, y, w, h = face

        # Safely clamp ROI boundaries
        offset = 5
        y1 = max(0, y - offset)
        y2 = min(frame.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(frame.shape[1], x + w + offset)

        face_section = frame[y1:y2, x1:x2]
        face_section = cv2.resize(face_section, (100, 100))

        if len(face_data) < max_samples:
            face_data.append(face_section.flatten())

        # Draw rectangle and sample count on screen
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {len(face_data)}/{max_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Collecting Face Data - Press 'q' to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if len(face_data) >= max_samples:
        print(f"Collected {max_samples} samples. Done!")
        break

cap.release()
cv2.destroyAllWindows()

# Save the face data to a .npy file
if face_data:
    face_data = np.array(face_data)
    save_path = os.path.join(dataset_path, f"{name}.npy")
    np.save(save_path, face_data)
    print(f"Face data saved to '{save_path}' with shape {face_data.shape}")
else:
    print("No face data collected. Make sure your face is visible to the camera.")
