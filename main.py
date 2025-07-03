import cv2
import numpy as np
from collections import namedtuple
import os
from datetime import datetime

# type alias for 2D coordinates (see create_sunglasses)
Point = namedtuple('Point', ['x', 'y'])

class FaceOverlay:
    """handles face detection and overlay application"""

    def __init__(self):
        # init pre-trained haar cascade models from openCV
        classifier_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(classifier_path)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load face cascade classifier")
            
        # overlay states
        self.show_overlay = False
        self.use_custom = False
        self.custom_overlay = None
    
    def detect_faces(self, frame):
        # detect faces using classifier initialized with the class
        #   this model works with grayscale images, so we convert the 
        #   incoming frames to grayscale first!
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.3, 5)
    
    def create_sunglasses(self, w, h):
        """lo-fi sunglasses overlay"""
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
    
    def apply_overlay(self, frame, x, y, w, h):
        """apply overlay to region of interest"""
        if not self.show_overlay:
            return
            
        if self.use_custom and self.custom_overlay is not None:
            # rescales the custom overlay depending on face bounding box
            overlay = cv2.resize(self.custom_overlay, (w, h))
            # mask any pixel that is non-white
            #   since we set the canvas background to white
            mask = np.any(overlay != [255, 255, 255], axis=2)
        else:
            # sunglasses overlay
            overlay = self.create_sunglasses(w, h)
            mask = np.any(overlay != [0, 0, 0], axis=2)
        
        # finalize
        roi = frame[y:y+h, x:x+w]
        roi[mask] = overlay[mask]

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

class PhotoBooth:
    """Main application class for Face Photo Booth"""
    
    def __init__(self):
        self.face_overlay = FaceOverlay()
        # for rendering visual text feedback when saving a photo
        self.save_message = ""
        self.save_message_timer = 0
        
        # init photos dir if needed
        self.photos_dir = "photos"
        os.makedirs(self.photos_dir, exist_ok=True)
    
    def save_photo(self, frame):
        """Save the current frame as a photo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.jpg"
        filepath = os.path.join(self.photos_dir, filename)
        
        try:
            cv2.imwrite(filepath, frame)
            self.save_message = f"Photo saved as: {filename}"
            self.save_message_timer = 50
            print(f"Photo saved to: {filepath}")
        except Exception as e:
            self.save_message = "Error saving photo!"
            self.save_message_timer = 50
            print(f"Error saving photo: {e}")
    
    def draw_ui(self, frame, faces):
        """Draw UI elements on the frame"""

        # count faces, if many
        cv2.putText(frame, f"Faces: {len(faces)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # to show current mode
        mode_text = "Mode: "
        if not self.face_overlay.show_overlay:
            mode_text += "No Overlay"
        elif self.face_overlay.use_custom:
            mode_text += "Custom Overlay"
        else:
            mode_text += "Sunglasses"
        cv2.putText(frame, mode_text, 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # save photo visual feedback (text confirmation)
        if self.save_message_timer > 0:
            cv2.putText(frame, self.save_message, 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            self.save_message_timer -= 1
    
    def run(self):
        """Main application loop"""
        # init camera feed, defaults to webcam/primary camera connected
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        print("Face Photo Booth Started!")
        print("Controls:")
        print("- Press SPACE to toggle overlay")
        print("- Press 'c' to create custom overlay")
        print("- Press 's' to save photo")
        print("- Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read from camera")
                    break
                
                # flip the feed, assuming front-facing webcam
                frame = cv2.flip(frame, 1)
                
                # detect faces and apply overlays if applicable
                faces = self.face_overlay.detect_faces(frame)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    self.face_overlay.apply_overlay(frame, x, y, w, h)
                
                # text UI
                self.draw_ui(frame, faces)
                
                # show processed/final frame
                cv2.imshow('Face Photo Booth', frame)
                
                # user controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): # quit
                    break
                elif key == ord(' '): # toggle overlay
                    if self.face_overlay.use_custom and not self.face_overlay.show_overlay:
                        self.face_overlay.use_custom = False
                    self.face_overlay.show_overlay = not self.face_overlay.show_overlay
                    print(f"Overlay {'enabled' if self.face_overlay.show_overlay else 'disabled'}")
                elif key == ord('s'): # snap/save frame
                    self.save_photo(frame)
                elif key == ord('c'): # open custom overlay window
                    new_overlay = create_custom_overlay()
                    if new_overlay is not None:
                        self.face_overlay.custom_overlay = new_overlay
                        self.face_overlay.use_custom = True
                        self.face_overlay.show_overlay = True
                        print("Custom overlay created and activated!")
                    else:
                        print("Custom overlay creation cancelled")
        
        finally:
            # cleanup
            cap.release()
            cv2.destroyAllWindows()

def main():
    """Application entry point"""
    try:
        booth = PhotoBooth()
        booth.run()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have a camera connected")
        print("2. Check if OpenCV is installed:")
        print("   pip install -r requirements.txt")
        print("3. Try closing other applications using the camera")

if __name__ == "__main__":
    main()
