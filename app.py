import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
from deepface import DeepFace

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('face_index.html', result=None)

@app.route('/analyze', methods=['POST'])
def analyze_face():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            img = cv2.imread(filepath)
            
            # Use MediaPipe backend for stability
            analysis = DeepFace.analyze(img, 
                                        actions=['age', 'gender', 'emotion'], 
                                        detector_backend='mediapipe',
                                        enforce_detection=True)
            
            face_data = analysis[0]
            result = {
                "image_path": filepath,
                "gender": face_data['dominant_gender'],
                "age": int(face_data['age']),
                "emotion": face_data['dominant_emotion'],
                "status": "Success"
            }
            
            # Draw a bounding box for user confirmation
            region = face_data['region']
            cv2.rectangle(img, (region['x'], region['y']), 
                          (region['x'] + region['w'], region['y'] + region['h']), (0, 255, 0), 4)
            cv2.imwrite(filepath, img)

        except Exception as e:
            result = {
                "image_path": filepath,
                "status": "Failed",
                "error": "No clear face detected. Please ensure good lighting."
            }

        return render_template('face_index.html', result=result)

if __name__ == '__main__':
    # Bind to port 10000 for production standard
    app.run(host='0.0.0.0', port=10000)