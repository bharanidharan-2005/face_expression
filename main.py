import cv2
import time
from collections import deque
from deepface import DeepFace
from stream_utils import LiveWebcam

# Initialize camera
cam = LiveWebcam(src=0).start()

# --- PROFESSIONAL ADDITIONS: TRACKING & SMOOTHING ---
frame_count = 0
skip_frames = 5  # Only run AI on every 5th frame (Huge speed boost!)
last_results = []

# Queues to hold the last 10 predictions for mathematical smoothing
age_history = deque(maxlen=10)
gender_history = deque(maxlen=10)
emotion_history = deque(maxlen=10)

# Performance variables
fps = 0
prev_time = 0

print("Initializing System... Press 'q' to safely terminate.")

while True:
    frame = cam.read()
    if frame is None:
        continue
        
    frame_count += 1
    current_time = time.time()

    # --- PROCESS ONLY ON CHOSEN FRAME INTERVAL ---
    if frame_count % skip_frames == 0:
        # Downscale for 4x faster processing speed
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        try:
            # Using MediaPipe backend for vastly superior accuracy under indoor lighting
            results = DeepFace.analyze(small_frame, 
                                      actions=['age', 'gender', 'emotion'],
                                      detector_backend='mediapipe',
                                      enforce_detection=False)
            last_results = results
        except Exception:
            pass

    # --- RENDER GRAPHICS & APPLY HISTORY SMOOTHING ---
    if last_results:
        for res in last_results:
            # Scale tracking coordinates back up to original frame dimensions
            x, y, w, h = [v * 2 for v in [res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']]]
            
            # Append current frame metrics to histories
            age_history.append(int(res['age']))
            gender_history.append(res['dominant_gender'])
            emotion_history.append(res['dominant_emotion'])

            # Calculate mathematical averages/modes to eliminate screen flickering
            smoothed_age = int(sum(age_history) / len(age_history))
            smoothed_gender = max(set(gender_history), key=gender_history.count)
            smoothed_emotion = max(set(emotion_history), key=emotion_history.count)

            # Draw UI Elements
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{smoothed_gender}, {smoothed_age}yrs, {smoothed_emotion}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- CALCULATE & DISPLAY METRIC FPS ---
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time
    cv2.putText(frame, f"System Engine FPS: {int(fps)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display stream window
    cv2.imshow("Live Professional Face Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()