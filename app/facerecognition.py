# Recognize faces with face_recognition library, which uses dlib underneath

import face_recognition
import argparse

def face_compare(image1Path: str, image2Path: str):
    img1 = face_recognition.load_image_file(image1Path)
    img2 = face_recognition.load_image_file(image2Path)
    img1encoding = face_recognition.face_encodings(img1)[0]
    img2encoding = face_recognition.face_encodings(img2)[0]
    result = face_recognition.compare_faces([img1encoding], img2encoding)
    print(result)

# MAIN EXECUTION
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare 2 faces images.')
    parser.add_argument('--image1', type=str, required=True)
    parser.add_argument('--image2', type=str, required=True)
    args = vars(parser.parse_args())
    print(f'Analyzing images')
    face_compare(args['image1'], args['image2'])
