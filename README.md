# Face Photo Booth App

A fun interactive photo booth application that will eventually support face detection and various overlays.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Current Features (v0.2)
- Real-time webcam feed
- Face detection with rectangle overlay
- Face count display
- Press 'q' to quit

## Coming Soon
- Fun overlays
- Photo saving
- Custom overlay creation

## Technical Details
The application uses OpenCV's Haar Cascade Classifier for face detection. The classifier is pre-trained and comes bundled with OpenCV. 