# Recognize faces with face_recognition library, which uses dlib underneath

import face_recognition
import argparse

def face_recognize(imageFilePath: str):
    image = face_recognition.load_image_file(imageFilePath)
    # Find all the faces in the image
    face_locations = face_recognition.face_locations(image)

    # Or maybe find the facial features in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    # Or you could get face encodings for each face in the image:
    list_of_face_encodings = face_recognition.face_encodings(image)

    print(f"Found {len(list_of_face_encodings)} faces in the image.")


# MAIN EXECUTION
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect faces in an image')
    parser.add_argument('--image', type=str, required=True)
    args = vars(parser.parse_args())
    print(f'Analyzing image {args["image"]}')
    face_recognize(args['image'])
