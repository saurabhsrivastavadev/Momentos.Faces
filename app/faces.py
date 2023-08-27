# Script to identify all faces in all image files in a folder.
# And prepare a database with all info.

import argparse
import imghdr
import os
from attr import dataclass
import cv2

from facedetect import face_detect
from numpy import ndarray

import face_recognition

@dataclass
class FacesInfo:
    """Data class for faces already processed, and currently in processing."""
    facesToProcess: list[ndarray]
    existingFaces: list[ndarray]
    newFaces: list[ndarray]

def update_faces(facesInfo: FacesInfo) -> bool:
    """Process the faces info and add any new faces to the object"""
    if facesInfo.facesToProcess.count == 0:
        return True
    if facesInfo.existingFaces.count == 0:
        facesInfo.newFaces.extend(facesInfo.facesToProcess)
        return True
    for faceToProcess in facesInfo.facesToProcess:
        for existingFace in facesInfo.existingFaces:
            img1encoding = face_recognition.face_encodings(faceToProcess)[0]
            img2encoding = face_recognition.face_encodings(existingFace)[0]
            result = face_recognition.compare_faces([img1encoding], img2encoding)
            print(f"face match: {result}")
            if result is False:
                facesInfo.newFaces.append(faceToProcess)
    return True

# MAIN EXECUTION
if __name__ == '__main__':
    # parse cli arguments 
    parser = argparse.ArgumentParser(description="Detect faces in an image or in all images at a path.")
    parser.add_argument('--image', type=str, required=False,
                        help="Process a single image")
    parser.add_argument('--path', type=str, required=False,
                        help="Process all images in the path")
    parser.add_argument('--appFolder', type=str, required=False, default='C:/Momentos',
                        help='Folder where all Momentos app meta data is saved, defaults to C:/Momentos')
    parser.add_argument('--skipprocessed', type=bool, required=False, default=True, 
                        help='Skip images already processed in earlier runs, already present in the app db')
    args = vars(parser.parse_args())

    # Validate args
    if args['image'] is not None and args['path'] is not None:
        print(f"Both image and path args can't be specified together.")
        exit()
    if args['image'] is None and args['path'] is None:
        print(f"Atleast one of image and path args must be specified.")
        exit()    

    # If image specified 
    if args["image"] is not None:
        imageFilePath = args["image"]
        print(f"Analyzing image {imageFilePath}")
        result = face_detect(imageFilePath=imageFilePath)
        print(f"Found {result.faceCount} faces in the image.")
        facesInfoToProcess = FacesInfo(
            existingFaces=[cv2.imread('c:/temp/face1.jpg')], 
            facesToProcess=[cv2.imread('c:/temp/face1_1.jpg')],
            newFaces=[])
        update_faces(facesInfoToProcess)
        print(facesInfoToProcess.newFaces)
        exit()

    # if path for images specified 
    if args["path"] is not None:
        imagesPath = args["path"]
        print(f"Analyzing images at path {imagesPath}")
        totalFiles = []
        imageFiles = []
        filesParseFailed = []
        fileCount = 0
        for (root, dirs, files) in os.walk(imagesPath):
            for file in files:
                fileCount += 1
        print(f"Found total {fileCount} files in {imagesPath}")
        parsedFileCount = 0
        for (root, dirs, files) in os.walk(imagesPath):
            for file in files:
                filePath = os.path.join(root, file)
                totalFiles.append(file)
                imgFileType = None
                try:
                    imgFileType = imghdr.what(filePath)
                except:
                     print(f"failed to parse file {file}")
                     filesParseFailed.append(file)
                if imgFileType is not None:
                    imageFiles.append(filePath)
                print(f"\rParsed {parsedFileCount} files out of {fileCount}", end="")
                parsedFileCount += 1

        print()
        print(f"Got {len(imageFiles)} image files in {imagesPath}")
        if len(filesParseFailed) > 0: 
            print(f"Failed to parse {len(filesParseFailed)} files")
        exit()

    print("No image or images folder specified!\n")
    parser.print_help()

