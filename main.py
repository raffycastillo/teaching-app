import cv2

def main():
    # initialize pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        return

    # capture inits
    cap = cv2.VideoCapture(0)
    print("Face Photo Booth Started!")
    print("Press 'q' to quit")
    
    # video feed loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera")
            break
            
        # flip and display the webcam feed
        frame = cv2.flip(frame, 1)
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # draw bounding boxes around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
        # there can be multiple faces detected,
        #   so we added a count.
        cv2.putText(frame, f"Faces: {len(faces)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # render the frame including all overlays
        cv2.imshow('Face Photo Booth', frame)
        
        # 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a camera connected and OpenCV installed:")
        print("pip install opencv-python")
