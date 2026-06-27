import mediapipe as mp
import cv2

IMAGE_NAME = 'thumb_up.png'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the image mode:
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.IMAGE)
with GestureRecognizer.create_from_options(options) as recognizer:

    webCam = cv2.VideoCapture(0)

    while True:
        b,frame = webCam.read()
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)

        # image = mp.Image.create_from_file(IMAGE_NAME)
        result = recognizer.recognize(image)
        if result.gestures:
            gestureName = result.gestures[0][0].category_name

            cv2.putText(frame, gestureName, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Image', frame)
        cv2.waitKey(1)