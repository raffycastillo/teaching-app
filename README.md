# Face Photo Booth

A fun interactive camera application that lets you add overlays to detected faces in real-time.

## Features

- Real-time face detection using OpenCV
- Built-in sunglasses overlay
- Custom overlay creation with mouse drawing
- Photo saving with timestamp-based filenames
- Support for multiple faces simultaneously

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Webcam/Camera device

## Installation

1. Clone this repository:
```bash
git clone https://github.com/raffycastlee/capstone/teaching-app.git
cd teaching-app
```

2. Create and activate a virtual environment (recommended):
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python main.py
```

By default, the application uses your system's default camera (usually the built-in webcam on laptops). If you have multiple cameras and want to use a different one, you can modify the camera index in `main.py`:
```python
# In main.py, change 0 to 1 for second camera, 2 for third camera, etc.
cap = cv2.VideoCapture(0)  # 0 is default camera
```

### Controls

- **SPACE**: Toggle overlay (switches between no overlay, sunglasses, custom)
- **C**: Open custom overlay creator
- **S**: Save current photo
- **Q**: Quit application

### Custom Overlay Creator

1. Press 'C' to open the drawing interface
2. Draw your design using the mouse (click and drag)
3. Press ENTER to save and use your overlay
4. Press ESC to cancel without saving

### Photos

- All photos are saved in the `photos` directory
- Filenames include timestamps (format: `photo_YYYYMMDD_HHMMSS.jpg`)
- Photos include any active overlays

## Troubleshooting

If you encounter issues:

1. Make sure your camera is connected and not in use by other applications
2. Verify OpenCV is installed correctly: `pip install -r requirements.txt`
3. Try closing other applications that might be using the camera
4. Ensure you have write permissions in the application directory for saving photos

## Technical Details

- Uses Haar cascade classifier for face detection
- Supports real-time overlay scaling based on face size
- Implements non-destructive overlay application
- Custom overlays use white as the transparent color 