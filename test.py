'''OpenCV documentation:
https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
'''
import cv2

'''Numpy Documentation:
https://numpy.org/doc/1.26/user/absolute_beginners.html
'''
import numpy as np

'''SKlearn documentation, purely for model training and testing.
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
'''
#for model training
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#test examples
'''To be filled.'''

'''Method to detect the skin in the camera frame'''
def detect_skin(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 50, 50], dtype = "uint8")
    upper = np.array([30, 255, 255], dtype = "uint8") #play with these values my needs
    skinMask = cv2.inRange(hsv, lower, upper)

    #Erotions and dilations to the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    #remove noise by blurring
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    #extract skin region
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    return skin

'''Finding Contours to identify hands and fingers.'''
def find_contours(skin):
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

'''Boxes drawn around hands to display contours.'''
def draw_contours(frame, contours):
    for contour in contours:
        if cv2.contourArea(contour) > 5000: #avoid noise by filtering small contours
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #rectangle shape

'''Classification of features from contours and contour area.'''
def extract_features(contours):
    features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        features.append([area, solidity])

    return features

''' Training the model.
def training():
    x = []
    y = []
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)

    print("Accuracy: ", accuracy_score(y_test, predicted))
'''

#initialize webcam
cap = cv2.VideoCapture(0) #0 is the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    skin = detect_skin(frame)
    contours = find_contours(skin)
    draw_contours(frame, contours)

    #Display capture frame
    cv2.imshow("Train my model!", frame)

    #Break loop with ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
