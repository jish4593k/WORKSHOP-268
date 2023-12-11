import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from tensorflow import keras

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

def load_regression_model(X_train, y_train):
    model, _ = stats.linregress(X_train, y_train)
    return model

def load_neural_network(X_train, y_train):
    input_size = X_train.shape[1]
    model = NeuralNetwork(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1000):
        inputs = torch.tensor(X_train.values, dtype=torch.float32)
        targets = torch.tensor(y_train.values, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    return model



def calculate_regression_prediction_scipy(model, face_encoding):
    return model.slope * face_encoding + model.intercept

def calculate_regression_prediction_torch(model, face_encoding):
    X_regression_nn = torch.tensor([face_encoding], dtype=torch.float32)
    return model(X_regression_nn).detach().numpy()[0][0]

class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition and Regression App")

        self.face_model = keras.models.load_model('face_model.h5')  # Replace with actual face recognition model

      
        data = pd.read_csv("your_regression_data.csv")
        y_regression = data.pop("target_column")
        X_regression_train, _, y_regression_train, _ = train_test_split(data, y_regression, test_size=0.2, random_state=42)

        self.scipy_model = load_regression_model(X_regression_train, y_regression_train)
        self.torch_model = load_neural_network(X_regression_train, y_regression_train)

        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.root, text="Select Image:").pack(pady=10)

        self.image_label = ttk.Label(self.root)
        self.image_label.pack()

        ttk.Button(self.root, text="Browse", command=self.browse_image).pack(pady=10)

        ttk.Label(self.root, text="Regression Prediction (SciPy):").pack(pady=5)
        self.scipy_label = ttk.Label(self.root)
        self.scipy_label.pack()

        ttk.Label(self.root, text="Regression Prediction (PyTorch):").pack(pady=5)
        self.torch_label = ttk.Label(self.root)
        self.torch_label.pack()

    def browse_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            image = Image.open(file_path)
            image = ImageTk.PhotoImage(image)

            self.image_label.config(image=image)
            self.image_label.image = image

            
            face_encoding = recognize_face(cv2.imread(file_path))

            
            regression_prediction_scipy = calculate_regression_prediction_scipy(self.scipy_model, face_encoding)
            self.scipy_label.config(text=f"SciPy: {regression_prediction_scipy:.2f}")

           
            regression_prediction_torch = calculate_regression_prediction_torch(self.torch_model, face_encoding)
            self.torch_label.config(text=f"PyTorch: {regression_prediction_torch:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    root.mainloop()
