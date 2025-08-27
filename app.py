from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import json
from datetime import datetime
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import io
import cv2  # Import OpenCV for video frame processing

app = Flask(__name__)
CORS(app)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# --- Config ---
IMAGE_MODEL_PATH = "image_model_output/ai_vs_human.keras"
DEEPFAKE_MODEL_PATH = "deepfake_detector_model.keras"  # Define path for video model
CLASS_NAMES_FILE = "class_names.json"
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024 # 100MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

# --- Load Class Names ---
if os.path.exists(CLASS_NAMES_FILE):
    with open(CLASS_NAMES_FILE, "r") as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = ["AI-generated", "human-created"]  # fallback
print(f"Loaded class names: {CLASS_NAMES}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image_model():
    if os.path.exists(IMAGE_MODEL_PATH):
        model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
        print(f"Image model loaded from {IMAGE_MODEL_PATH}")
        return model
    return None

def load_deepfake_model():
    if os.path.exists(DEEPFAKE_MODEL_PATH):
        try:
            model = tf.keras.models.load_model(DEEPFAKE_MODEL_PATH)
            print(f"Deepfake model loaded from {DEEPFAKE_MODEL_PATH}")
            return model
        except Exception as e:
            print(f"Error loading deepfake model: {e}")
    return None

# Initialize models
image_model = load_image_model()
deepfake_model = load_deepfake_model()

def preprocess_image(image):
    """Preprocess image to match training normalization."""
    img = Image.open(io.BytesIO(image)).convert('RGB')
    img = img.resize((160, 160))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Rescale to [0, 1]
    return np.expand_dims(img_array, axis=0)
    
def preprocess_video(video_file):
    """
    Extracts and preprocesses frames from a video.
    """
    frames_list = []
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    while frame_count < 100 and cap.isOpened(): 
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        rgb_frame = rgb_frame / 255.0
        frames_list.append(rgb_frame)
        frame_count += 1
        
    cap.release()
    if len(frames_list) == 0:
        return None
    return np.array(frames_list)

# --- Image Routes (UPDATED) ---
@app.route('/analyze/image', methods=['POST'])
def analyze_image():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file type"}), 400
        
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > MAX_IMAGE_SIZE:
            return jsonify({"success": False, "error": "File too large"}), 400
        
        if image_model is None:
            return jsonify({"success": False, "error": "Image model not loaded"}), 503
        
        img_array = preprocess_image(file.read())
        prediction = image_model.predict(img_array)
        raw_score = float(prediction[0][0])
        
        # Correct sigmoid interpretation
        human_prob = raw_score * 100
        predicted_class = CLASS_NAMES[1] if raw_score >= 0.5 else CLASS_NAMES[0]
        
        return jsonify({
            "success": True,
            "type": "image",
            "score": round(human_prob, 2),
            "raw_score": round(raw_score, 4),
            "analysis": f"This image appears {predicted_class} ({round(human_prob,2)}% human probability)",
            "model_version": "image_model_v1",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Image analysis error: {str(e)}"}), 500

# --- Video Routes (NEW) ---
@app.route('/analyze/video', methods=['POST'])
def analyze_video():
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file type"}), 400

        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > MAX_VIDEO_SIZE:
            return jsonify({"success": False, "error": "File too large"}), 400

        if deepfake_model is None:
            return jsonify({"success": False, "error": "Video model not loaded"}), 503

        # Save the file temporarily to process it with OpenCV
        temp_path = "temp_video.mp4"
        file.save(temp_path)

        processed_frames = preprocess_video(temp_path)
        os.remove(temp_path)  # Clean up the temp file
        
        if processed_frames is None or processed_frames.shape[0] == 0:
            return jsonify({"success": False, "error": "Failed to process video frames"}), 500

        # Run prediction on the processed frames
        prediction = deepfake_model.predict(processed_frames)
        # Assuming the model returns a score for each frame.
        # Average the scores for a final video score.
        raw_score = float(np.mean(prediction))

        # Assuming the model outputs a value between 0 and 1, where a higher value indicates human-created.
        human_prob = raw_score * 100
        predicted_class = CLASS_NAMES[1] if raw_score >= 0.5 else CLASS_NAMES[0]

        return jsonify({
            "success": True,
            "type": "video",
            "score": round(human_prob, 2),
            "raw_score": round(raw_score, 4),
            "analysis": f"This video appears {predicted_class} ({round(human_prob, 2)}% human probability)",
            "model_version": "deepfake_model_v1",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Video analysis error: {str(e)}"}), 500

# --- Model Status Route (UPDATED) ---
@app.route('/model/status', methods=['GET'])
def model_status():
    return jsonify({
        "image_model_loaded": image_model is not None,
        "video_model_loaded": deepfake_model is not None,
        "image_model_version": "image_model_v1",
        "video_model_version": "deepfake_model_v1",
        "class_names": CLASS_NAMES
    })

@app.route('/')
def home():
    return render_template('index.html',
                           image_model_status=image_model is not None,
                           video_model_status=deepfake_model is not None)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)