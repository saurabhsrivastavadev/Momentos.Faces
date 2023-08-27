# Script to return faces in the specified image
# Uses MediaPipe library
# https://google.github.io/mediapipe/solutions/face_detection#min_detection_confidence

from array import ArrayType
from dataclasses import dataclass
from pydoc import describe
from PIL import Image
import cv2
import mediapipe as mp

from numpy import ndarray

# Data Types 
@dataclass
class FaceDetectionResult:
    """Class to share face detection result with client"""
    success: bool
    faceCount: int
    croppedFaces: list[ndarray]

# FUNCTION - FACE_DETECT
def face_detect(imageFilePath: str, model_selection: int = 1, 
                min_detection_confidence: float = 0.5) -> FaceDetectionResult:
    """
    Detect and return list of faces in an image.

    Args:
        MODEL_SELECTION: An integer index 0 or 1. Use 0 to select a short-range model that works
                         best for faces within 2 meters from the camera, and 1 for a full-range 
                         model best for faces within 5 meters. For the full-range option, a sparse 
                         model is used for ts improved inference speed. Please refer to the model 
                         cards for details. 
                         Default to 0 if not specified.
        MIN_DETECTION_CONFIDENCE: Minimum confidence value ([0.0, 1.0]) from the face detection 
                                  model for the detection to be considered successful. 
                                  Default to 0.5.

    Returns:
        DETECTIONS: Collection of detected faces, where each face is represented as a detection 
                    proto message that contains a bounding box and 6 key points (right eye, 
                    left eye, nose tip, mouth center, right ear tragion, and left ear tragion). 
                    The bounding box is composed of xmin and width (both normalized to [0.0, 1.0] 
                    by the image width) and ymin and height (both normalized to [0.0, 1.0] by the 
                    image height). Each key point is composed of x and y, which are normalized to 
                    [0.0, 1.0] by the image width and height respectively.
    """
    print("face_detect")
    mpFaceDetection = mp.solutions.face_detection # type: ignore
    mpDrawing = mp.solutions.drawing_utils # type: ignore

    with mpFaceDetection.FaceDetection(model_selection, min_detection_confidence) as faceDetection:

        image = cv2.imread(imageFilePath)
        results = faceDetection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.detections:
            return FaceDetectionResult(success=False, faceCount=0, croppedFaces=[])

        faces:list[ndarray] = []
        for detection in results.detections:
            print(detection)
            height, width, channels = image.shape
            croppedHeight = (int)(detection.location_data.relative_bounding_box.height * height)
            croppedWidth = (int)(detection.location_data.relative_bounding_box.width * width)

            # 120 px margin around the face for extraction
            croppedXMin = (int)(detection.location_data.relative_bounding_box.xmin * width) - 120
            croppedYMin = (int)(detection.location_data.relative_bounding_box.ymin * height) - 120
            croppedXMax = croppedXMin + croppedWidth + 120
            croppedYMax = croppedYMin + croppedHeight + 120

            # Adjust margins
            if croppedXMin < 0:
                croppedXMin = 0
            if croppedXMax > width:
                croppedXMax = width - 5
            if croppedYMin < 0:
                croppedYMin = 0
            if croppedYMax > height:
                croppedYMax = height - 5

            croppedImage:ndarray = image[croppedYMin:croppedYMax, croppedXMin:croppedXMax]
            faces.append(croppedImage)

        return FaceDetectionResult(success=True, faceCount=len(results.detections), croppedFaces=faces)
