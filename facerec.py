import numpy as np
import dlib
import cv2
import os
import PIL.Image
import math
import operator

predictor_path = "/usr/src/app/haar/shape_predictor_68_face_landmarks.dat" 
rootdir = '/usr/src/app/images'
font = cv2.FONT_HERSHEY_SIMPLEX
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

recognizer = cv2.face.createLBPHFaceRecognizer()

images = []
labels = []
labels_index = []

def getFaces(frame):
    dets = detector(frame, 1)
    return dets

def highlightFaces(frame, faces):
    for face in faces:
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255,255,255), 1, 0)
    return frame

def markFace(frame, shape):
    cv2.circle(frame, (shape.parts()[0].x, shape.parts()[0].y), 2, (255,255,255), -1)
    cv2.circle(frame, (shape.parts()[8].x, shape.parts()[8].y), 2, (255,255,255), -1)
    cv2.circle(frame, (shape.parts()[16].x, shape.parts()[16].y), 2, (255,255,255), -1)
    cv2.circle(frame, (shape.parts()[18].x, shape.parts()[18].y), 2, (255,255,255), -1)
    cv2.circle(frame, (shape.parts()[24].x, shape.parts()[24].y), 2, (255,255,255), -1)
    cv2.circle(frame, (shape.parts()[30].x, shape.parts()[30].y), 2, (255,255,255), -1)
    cv2.circle(frame, (shape.parts()[36].x, shape.parts()[36].y), 2, (255,255,255), -1)
    cv2.circle(frame, (shape.parts()[39].x, shape.parts()[39].y), 2, (255,255,255), -1)
    cv2.circle(frame, (shape.parts()[42].x, shape.parts()[42].y), 2, (255,255,255), -1)
    cv2.circle(frame, (shape.parts()[45].x, shape.parts()[45].y), 2, (255,255,255), -1)
    return frame

def cropFace(frame, shape):
    #    points = shape.parts()
#    y1 = points[18].y if points[18].y < points[24].y else points[24].y
#    y2 = points[8].x
#    x1 = points[0].x
#    x2 = points[16].x
#
#    face = frame[y1:y2, x1:x2]
    face = frame[0:200, 0:200]
    face = cv2.resize(face, (200, 200))
    #cv2.imshow("face", face)
    return face

def centerPoint(p1, p2):
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    return (x1 + (x2 - x1)* .5, y1 + (y2 - y1) * .5)

# returns the center position of right eye as tuple
def rightEye(shape):
    p1 = shape.parts()[36]
    p2 = shape.parts()[39]
    return centerPoint(p1, p2)

# returns the center position of right eye as tuple
def leftEye(shape):
    p1 = shape.parts()[42]
    p2 = shape.parts()[46]
    return centerPoint(p1, p2)

# Find angle
def faceAngle(shape):
    p1 = shape.parts()[0]
    p2 = shape.parts()[16]
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    dx,dy = x2-x1,y2-y1

    rads = -math.atan2(dy, dx)
    return math.degrees(rads)

#rotate image around a point
def rotateImage(image, point, angle):
    rows,cols = image.shape 
    M = cv2.getRotationMatrix2D(point,angle * -1,1)
    return cv2.warpAffine(image, M, (cols,rows))

def affineTransform(image, shape):
    points = shape.parts()
    rows,cols = image.shape

    x1, y1 = points[36].x,  points[36].y
    x2, y2 = points[45].x, points[45].y

    x3, y3 = points[8].x, points[8].y 


    pts1 = np.float32([[x3,y3],[x1,y1],[x2,y2]])
    pts2 = np.float32([[100,200],[0,70],[200,70]])

    M = cv2.getAffineTransform(pts1,pts2)

    return cv2.warpAffine(image, M,(cols,rows))

def main():
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            image = cv2.imread(os.path.join(subdir, file), 0)
            faces = getFaces(image)
            if len(faces) > 0:
                for face in faces:
                    shape = predictor(image, face)
                    #angle = faceAngle(shape)
                    #righteye = rightEye(shape)

                    rotatedFace = affineTransform(image, shape)
                    faceImage = cropFace(rotatedFace, shape)
                    faceImage = clahe.apply(faceImage)

                    images.append(faceImage)
                    labels.append(subdir)
                    #cv2.imshow(subdir + file, faceImage)
    for i, value in enumerate(labels):
        labels_index.append(i)

    recognizer.train(images, np.array(labels_index))        

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        faces = getFaces(frame)
        if len(faces) > 0:
            for i, face in enumerate(faces):
                shape = predictor(frame, face)
                #angle = faceAngle(shape)
                #righteye = rightEye(shape)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                gray_frame = affineTransform(gray_frame, shape)
                faceImage = cropFace(gray_frame, shape)
                faceImage = clahe.apply(faceImage)
                cv2.imshow(str(i), faceImage)
                frame = markFace(frame, shape)

                label, conf = recognizer.predict(faceImage)
                confPercent = 100 - ((100/100) * float(conf))
                if (confPercent  > 30):
                    name = labels[label].split("/")[len(labels[label].split("/")) -1]
                    cv2.putText(frame, "L: " + name + " conf: " + str(format(confPercent, '.2f')), (face.left() + 30,face.top()), cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),2 )

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
