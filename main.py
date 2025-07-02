import cv2
import numpy as np
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])

def create_sunglasses(w, h):
    """sunglasses overlay for the provided region"""
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    
    # sunglasses total rectangle canvas
    #  about 1/3 of the face's width (horizontal)
    #  about 1/6 of the face's length (vertical)
    glass_w = round(w / 1.5)
    glass_h = h // 4
    # sunglasses top left coordinates:
    glass_tl = Point((w-glass_w) // 2, h // 4)

    # sunglasses widths by piece
    left_lens_w = right_lens_w = (glass_w // 5) * 2
    bridge_w = glass_w // 5
    # sunglasses heights by piece
    left_lens_h = right_lens_h = glass_h
    bridge_h = glass_h // 7
    # sunglasses' pieces start coordinates
    left_lens_tl = Point(glass_tl.x, glass_tl.y)
    right_lens_tl = Point(glass_tl.x + left_lens_w + bridge_w, glass_tl.y)
    bridge_tl = Point(glass_tl.x + left_lens_w, glass_tl.y + ((glass_h // 7) * 3))

    # left lens
    cv2.rectangle(overlay, left_lens_tl, (left_lens_tl.x+left_lens_w, left_lens_tl.y + left_lens_h), (61, 61, 61), -1)
    # right lens
    cv2.rectangle(overlay, right_lens_tl, (right_lens_tl.x+right_lens_w, right_lens_tl.y + right_lens_h), (61, 61, 61), -1)
    # middle bridge
    cv2.rectangle(overlay, bridge_tl, 
                 (bridge_tl.x+bridge_w, bridge_tl.y+bridge_h), (255, 255, 255), -1)
    
    return overlay

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
    print("Press SPACE to toggle sunglasses")
    
    # to track if filter is enabled
    show_sunglasses = False
    
    # video feed loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera")
            break
            
        # flip and display the webcam feed
        frame = cv2.flip(frame, 1)
        
        # convert feed to greyscale before processing
        #   the pre-trained classifier with openCV uses grayscale!
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # draw bounding boxes around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # apply overlay if applicable
            if show_sunglasses:
                # initialize the sunglass render
                glasses = create_sunglasses(w, h)
                # find the region of interest (roi)
                #   and overlay the sunglasses of the base feed
                roi = frame[y:y+h, x:x+w]
                mask = np.any(glasses != [0, 0, 0], axis=2)
                roi[mask] = glasses[mask]
        
        # there can be multiple faces detected,
        #   so we added a count.
        cv2.putText(frame, f"Faces: {len(faces)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # to show current mode
        mode_text = "Sunglasses: ON" if show_sunglasses else "Sunglasses: OFF"
        cv2.putText(frame, mode_text, 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # render the frame including all overlays
        cv2.imshow('Face Photo Booth', frame)
        
        # handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # quitting
            break
        elif key == ord(' '):  # spacebar toggles sunglasses
            show_sunglasses = not show_sunglasses
            print(f"Sunglasses {'enabled' if show_sunglasses else 'disabled'}")
    
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
