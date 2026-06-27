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
    image = mp.Image.create_from_file(IMAGE_NAME)
    results = recognizer.recognize(image)
    print(results.gestures[0][0].category_name, results.gestures[0][0].score)

    i = cv2.imread(IMAGE_NAME)
    resized_i = cv2.resize(i, (512, 512), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Image', resized_i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()