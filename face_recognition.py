import numpy as np
import cv2
import os


def distance(v1, v2):

    return np.sqrt(((v1-v2)**2).sum())


def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):

        ix = train[i, :-1]
        iy = train[i, -1]

        d = distance(test, ix)
        dist.append([d, iy])

    dk = sorted(dist, key=lambda x: x[0])[:k]

    labels = np.array(dk)[:, -1]

    output = np.unique(labels, return_counts=True)

    index = np.argmax(output[1])
    return int(output[0][index])


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
if face_cascade.empty():
    print("Error: Could not load haarcascade_frontalface_alt.xml. Make sure it's in the same folder.")
    exit()

dataset_path = "./face_dataset/"

face_data = []
labels = []
class_id = 0
names = {}

if not os.path.exists(dataset_path) or not any(f.endswith('.npy') for f in os.listdir(dataset_path)):
    print(
        f"Error: No face data found in '{dataset_path}'. Please run face_data.py first.")
    cap.release()
    exit()

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
print(face_labels.shape)
print(face_dataset.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if ret == False:
        print("Warning: Failed to grab frame. Camera may have disconnected.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for face in faces:
        x, y, w, h = face

# Get the face ROI
        offset = 5
        y1 = max(0, y - offset)
        y2 = min(frame.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(frame.shape[1], x + w + offset)
        face_section = frame[y1:y2, x1:x2]
        face_section = cv2.resize(face_section, (100, 100))

        out = knn(trainset, face_section.flatten())

# Draw rectangle in the original image
        cv2.putText(frame, names[out], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
