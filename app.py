import tensorflow as tf
import numpy as np
import cv2
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the TFLite model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.tflite')

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Labels and suggestions (no changes needed)
labels = ["Healthy", "Leaf Mold", "Late Blight", "Early Blight"]
suggestions = {
    "Healthy": "Your tomato plant is healthy! Keep up the good work.",
    "Leaf Mold": "Apply fungicides containing copper or chlorothalonil. Ensure proper ventilation and avoid overhead watering.",
    "Late Blight": "Remove infected leaves and apply fungicides like mancozeb. Avoid wet conditions.",
    "Early Blight": "Prune infected leaves and apply fungicides. Rotate crops annually."
}

@app.route('/')
def home():
    return jsonify({"message": "Tomato Disease Detection API is Running!"})

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    try:
        # Read and preprocess the image
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))  # Resize to model input size
        image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Run inference with TFLite
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Get results
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class])
        prediction = labels[predicted_class]
        suggestion = suggestions[prediction]

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "suggestion": suggestion
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
