# Script to return faces in the specified image
# Uses MediaPipe library
# https://google.github.io/mediapipe/solutions/face_detection#min_detection_confidence

from array import ArrayType
from dataclasses import dataclass
from pydoc import describe
import cv2
import mediapipe as mp
import argparse
import imghdr
import os

# Data Types 
@dataclass
class FaceDetectionResult:
    """Class to share face detection result with client"""
    success: bool
    faceCount: int

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
    # parse cli arguments 
    parser = argparse.ArgumentParser(description="Detect faces in an image or in all images at a path.")
    parser.add_argument('--image', type=str, help="Process a single image", required=False)
    parser.add_argument('--path', type=str, help="Process all images in the path", required=False)
    args = vars(parser.parse_args())

    # if image specified 
    if args["image"] is not None:
        imageFilePath = args["image"]
        print(f"Analyzing image {imageFilePath}")
        result = face_detect(imageFilePath=imageFilePath)
        print(f"Found {result.faceCount} faces in the image.")
        exit()

    # if path for images specified 
    if args["path"] is not None:
        imagesPath = args["path"]
        print(f"Analyzing images at path {imagesPath}")
        totalFiles = []
        imageFiles = []
        filesParseFailed = []
        for (root, dirs, files) in os.walk(imagesPath):
            for file in files:
                filePath = os.path.join(root, file)
                totalFiles.append(file)
                try:
                    imgFileType = imghdr.what(filePath)
                except:
                    filesParseFailed.append(file)

                if imgFileType is not None:
                    imageFiles.append(filePath)

        print(f"Got {len(totalFiles)} total files in {imagesPath}")
        print(f"Got {len(imageFiles)} image files in {imagesPath}")
        print(f"Failed to parse {len(filesParseFailed)} files")
        exit()

    print("No image or images path specified\n")
    parser.print_help()

