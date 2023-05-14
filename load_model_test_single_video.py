import torch
import cv2
import numpy as np
from scripts.prepare_lrw import extract_opencv
from google.colab import files

# Load the trained model
# model = YourModelClass()
# model.load_state_dict(torch.load('/path/to/weights/checkpoint.pth'))

#video_model = torch.load(checkpoint_path)

# Define a function to preprocess the video frames
def preprocess_frames(frames):
    preprocessed_frames = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (224, 224))
        frame = np.expand_dims(frame, axis=2)
        preprocessed_frames.append(frame)
    preprocessed_frames = np.array(preprocessed_frames)
    preprocessed_frames = np.transpose(preprocessed_frames, (3, 0, 1, 2))
    preprocessed_frames = preprocessed_frames.astype(np.float32) / 255.0
    return preprocessed_frames


# Get the uploaded video file
uploaded = files.upload()

# Get the video frames
filename = list(uploaded.keys())[0]
frames = extract_opencv(filename)

# Preprocess the frames and insert them into the model
frames = preprocess_frames(frames)
with torch.no_grad():
    predictions = model(frames)

# Get the predicted label
predicted_label = torch.argmax(predictions).item()

# Print the predicted label
print(f'Predicted label: {predicted_label}')