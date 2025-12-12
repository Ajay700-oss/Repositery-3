import cv2
from deepface import DeepFace

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face and emotion
    try:
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # DeepFace returns a list of dictionaries
        for result in results:
            emotion = result['dominant_emotion']
            face_box = result['region']  # Dictionary with x, y, w, h
            
            # Draw rectangle around face
            x, y, w, h = face_box['x'], face_box['y'], face_box['w'], face_box['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Put emotion text above the rectangle
            cv2.putText(frame, f'{emotion}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        pass

    # Show video feed
    cv2.imshow("Live Face & Emotion Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
