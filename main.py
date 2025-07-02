import cv2

def main():
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
