import cv2
import numpy as np
import tensorflow as tf
from google.colab.patches import cv2_imshow
# Load your trained TensorFlow FCN model
model = tf.keras.models.load_model("/content/drive/MyDrive/fully_convolutional_neural_network/model.h5")
  # Replace with your model's path

# Define video input and output
video_path = "/content/4832674-uhd_3840_2160_30fps (1).mp4"  # Replace with your video file
output_video = "segmented_traffic.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Function to preprocess a frame
def preprocess_frame(frame, target_size=(224, 224)):  # Change to model's expected size
    frame_resized = cv2.resize(frame, target_size)  # Resize to match model input
    frame_normalized = frame_resized / 255.0  # Normalize pixel values
    return np.expand_dims(frame_normalized, axis=0)  # Add batch dimension


# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_frame = preprocess_frame(frame)

    # Run inference
    prediction = model.predict(input_frame)[0]  # Get single-frame output
    segmentation_mask = np.argmax(prediction, axis=-1)  # Convert to class labels
    segmentation_mask = cv2.resize(segmentation_mask.astype(np.uint8), (frame_width, frame_height))

    # Apply color map to segmentation mask
    mask_colored = cv2.applyColorMap(segmentation_mask * 30, cv2.COLORMAP_JET)

    # Overlay mask on original frame
    blended = cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)

    # Write frame to output video
    out.write(blended)

    # Display result (optional)
    cv2_imshow(blended)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Segmented video saved as {output_video}")
