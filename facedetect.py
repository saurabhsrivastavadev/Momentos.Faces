# Script to return faces in the specified image
# Uses MediaPipe library
# https://google.github.io/mediapipe/solutions/face_detection#min_detection_confidence

from array import ArrayType
from pydoc import describe
import cv2
import mediapipe as mp
import argparse

# Data Types 
class FaceDetectionResult:
    def __init__(self, success:bool = False, faceCount:int = 0) -> None:
        self.success = success
        self.faceCount = faceCount

def face_detect(imageFilePath: str, model_selection: int = 1, 
                min_detection_confidence: float = 0.5) -> FaceDetectionResult:
    """
    Detect and return list of faces in an image.

    Args:
        MODEL_SELECTION: An integer index 0 or 1. Use 0 to select a short-range model that works
                         best for faces within 2 meters from the camera, and 1 for a full-range 
                         model best for faces within 5 meters. For the full-range option, a sparse 
                         model is used for its improved inference speed. Please refer to the model 
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
    mpFaceDetection = mp.solutions.face_detection
    mpDrawing = mp.solutions.drawing_utils

    with mpFaceDetection.FaceDetection(model_selection, min_detection_confidence) as faceDetection:
        image = cv2.imread(imageFilePath)
        results = faceDetection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.detections:
            return FaceDetectionResult(success=False, faceCount=0)
        
        annotatedImage = image.copy()
        for detection in results.detections:
            mpDrawing.draw_detection(annotatedImage, detection)

        return FaceDetectionResult(success=True, faceCount=len(results.detections))


# MAIN EXECUTION
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect faces in an image')
    parser.add_argument('--image', type=str, required=True)
    args = vars(parser.parse_args())
    print(f'Analyzing image {args["image"]}')
    result = face_detect(args['image'])
    print(f"Found {result.faceCount} faces in the image.")
