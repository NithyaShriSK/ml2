
import kagglehub
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


path = kagglehub.dataset_download("emmarex/plantdisease")
dataset_path = os.path.join(path, "PlantVillage", "PlantVillage")

X = []
y = []
classes = []


for label, folder in enumerate(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    classes.append(folder)

    for file in os.listdir(folder_path)[:200]:
        img = cv2.imread(os.path.join(folder_path, file))

        if img is not None:
            img = cv2.resize(img, (64, 64))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            features = cv2.mean(hsv)[:3]

            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
model = SVC(kernel="linear", class_weight="balanced")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Number of classes:", len(np.unique(y)))
print("Total images:", len(y))
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(
    y_test, y_pred,
    labels=np.unique(y),
    target_names=[classes[i] for i in np.unique(y)]
))
