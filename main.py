import cv2
import numpy as np
from collections import namedtuple
import os
from datetime import datetime
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

def create_custom_overlay():
    """draw your own overlay!"""
    # initiate white canvas, 300x400px window
    canvas = np.ones((300, 400, 3), dtype=np.uint8) * 255
    drawing = False
    last_point = None

    # callbacks defined for openCV MUST have these five params
    #   even if they're not being used.
    def draw(event, x, y, flags, param):
        nonlocal drawing, last_point
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(canvas, last_point, (x, y), (0, 0, 0), 2)
                last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            last_point = None

    # init new window and attach callback
    cv2.namedWindow('Custom Overlay Creator')
    cv2.setMouseCallback('Custom Overlay Creator', draw)

    # console instructions
    print("\nCustom Overlay Creator")
    print("- Draw your overlay using the mouse")
    print("- Press ENTER to save and use")
    print("- Press ESC to cancel\n")

    while True:
        cv2.imshow('Custom Overlay Creator', canvas)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # ENTER key
            cv2.destroyWindow('Custom Overlay Creator')
            return canvas
        elif key == 27:  # ESC key
            cv2.destroyWindow('Custom Overlay Creator')
            return None

def main():
    # initialize pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier")
        return

    # capture inits
    cap = cv2.VideoCapture(0)
    print("Face Photo Booth Started!")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press SPACE to toggle sunglasses")
    print("- Press 's' to save photo")
    print("- Press 'c' to create custom overlay")
    
    # to track if filter is enabled and which type
    show_overlay = False
    use_custom = False
    custom_overlay = None
    
    # init photos dir
    photos_dir = "photos"
    if not os.path.exists(photos_dir):
        os.makedirs(photos_dir)
    
    # for rendering visual text feedback
    #   when saving a photo
    save_message = ""
    save_message_timer = 0
    
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
            if show_overlay:
                if use_custom and custom_overlay is not None:
                    # rescales the custom overlay depending on face bounding box
                    overlay = cv2.resize(custom_overlay, (w, h))
                    # mask any pixel that is non-white
                    #   since we set the canvas background to white
                    mask = np.any(overlay != [255, 255, 255], axis=2)
                else:
                    # sunglasses overlay
                    overlay = create_sunglasses(w, h)
                    mask = np.any(overlay != [0, 0, 0], axis=2)
                
                # apply
                roi = frame[y:y+h, x:x+w]
                roi[mask] = overlay[mask]
        
        # there can be multiple faces detected,
        #   so we added a count.
        cv2.putText(frame, f"Faces: {len(faces)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # to show current mode
        mode_text = "Mode: "
        if not show_overlay:
            mode_text += "No Overlay"
        elif use_custom:
            mode_text += "Custom Overlay"
        else:
            mode_text += "Sunglasses"
        cv2.putText(frame, mode_text, 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # save photo visual feedback (text confirmation)
        if save_message_timer > 0:
            cv2.putText(frame, save_message, 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            save_message_timer -= 1
        
        # render the frame including all overlays
        cv2.imshow('Face Photo Booth', frame)
        
        # handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # quitting
            break
        elif key == ord(' '):  # spacebar toggles overlay
            if use_custom and not show_overlay:
                # If custom was last used, switch back to sunglasses
                use_custom = False
            show_overlay = not show_overlay
            print(f"Overlay {'enabled' if show_overlay else 'disabled'}")
        elif key == ord('s'):  # save photo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"photo_{timestamp}.jpg"
            filepath = os.path.join(photos_dir, filename)
            cv2.imwrite(filepath, frame)
            save_message = f"Photo saved as: {filename}"
            save_message_timer = 50  # Display message for 50 frames
            print(f"Photo saved to: {filepath}")
        elif key == ord('c'):  # create custom overlay
            new_overlay = create_custom_overlay()
            if new_overlay is not None:
                custom_overlay = new_overlay
                use_custom = True
                show_overlay = True
                print("Custom overlay created and activated!")
            else:
                print("Custom overlay creation cancelled")
    
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
