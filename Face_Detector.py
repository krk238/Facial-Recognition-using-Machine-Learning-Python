# Kamil Khan
# Face Detector Application
# This application detects faces of persons in photos and highlights them with a rectangle or square
# Pre - Trained Data is obtained from https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

import cv2

# --------------------------------------------------------------------------------------------------
# Face Detection in Images
# Algorithm - Haar Cascade
# --------------------------------------------------------------------------------------------------


def picture_Face_Detector(img_or_vid):
    # 1 - Load the faces picture (imread)
    img1 = cv2.imread(img_or_vid)

    # 2 - Convert to Grayscale color format (cvtColor)
    img1_gray_scaled = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # 3 - Load the pre-trained facial data (Cascade Classifier)
    trained_facial_data = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml")

    # 4 - Apply Trained Data to recognize faces and create coordinates for rectangle/square (detectMultiScale)
    facial_coordinates = trained_facial_data.detectMultiScale(img1_gray_scaled)

    # 5 - Apply the rectangle to the faces
    for crd_set in facial_coordinates:
        (x, y, w, h) = crd_set
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #6 - Show the original colored image with the faces squared
    cv2.imshow("Intelligent Face detector image", img1)
    cv2.waitKey()


# --------------------------------------------------------------------------------------------------
# Face Detection in real - time Video via webcam
# Algorithm - Haar Cascade
# --------------------------------------------------------------------------------------------------


def webcam_Face_Detector():
    # 1 - Loads the pre-trained facial data (Cascade Classifier)
    trained_facial_data = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml")

    # 2 - Loads the default webcam or video
    webCam = cv2.VideoCapture(0)

    # 3 - This is the apparatus to check each frame of the video for faces
    while webCam:
        # Reads the frames from the video
        frame_feedback, frames = webCam.read()

        # Greyscales the video
        frames_gray_scaled = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

        # Runs the haar features through the image and obtains coordinates of a fac
        facial_coordinates = trained_facial_data.detectMultiScale(
            frames_gray_scaled)

        # Goes through coordinate sets and creates a frame aruond the face
        for crd_set in facial_coordinates:
            (x, y, w, h) = crd_set
            cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Displays the video or webcam
        cv2.imshow("Intelligent Face Detector", frames)
        key = cv2.waitKey(1)

        # ASCII Exit key to stop the application (Q)
        if key == 81:
            break


webcam_Face_Detector()
