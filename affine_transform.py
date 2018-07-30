from __future__ import print_function
import numpy as np
import cv2
import dlib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--with_draw', help='do draw?', default='True')
args = parser.parse_args()

net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt', './models/res10_300x300_ssd_iter_140000.caffemodel')
landmark_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

# cv2.imshow("show", np.zeros((50,50,3)))
# cv2.waitKey(100)
# cv2.destroyAllWindows()

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def preprocess(img):
    ### analysis
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(3):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_img.mean() < 130:
            img = adjust_gamma(img, 1.5)
        else:
            break
    return img



cv2.namedWindow('show', 0)

idx_total = 0

bgr_img = cv2.imread('./image.png', 1)
if bgr_img is None:
    exit()

start = cv2.getTickCount()

img_origin = bgr_img.copy()
bgr_img = preprocess(bgr_img)

### detection
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
(h, w) = bgr_img.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(bgr_img, (300, 300)), 1.0,
    (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

### bbox
list_bboxes = []
list_confidence = []
list_dlib_rect = []
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence < 0.6:
            continue
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (l, t, r, b) = box.astype("int") # l t r b
    
    original_vertical_length = b-t
    t = int(t + (original_vertical_length)*0.15)
    b = int(b - (original_vertical_length)*0.05)

    margin = ((b-t) - (r-l))//2
    l = l - margin if (b-t-r+l)%2 == 0 else l - margin - 1
    r = r + margin
    list_bboxes.append([l, t, r, b])
    list_confidence.append(confidence)
    rect_bb = dlib.rectangle(left=l, top=t, right=r, bottom=b)
    list_dlib_rect.append(rect_bb)

### landmark
list_landmarks = []
for rect in list_dlib_rect:
    points = landmark_predictor(rgb_img, rect)
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)

time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
print ('elapsed time: %.2fms'%time)

for landmark in list_landmarks:
    
    pts_origin = np.float32([landmark[30], landmark[48], landmark[8], landmark[54]])

    pts_target = np.float32([[150,0], [0,150], [150,330], [300,150]])

    M = cv2.getPerspectiveTransform(pts_origin, pts_target)
    dst = cv2.warpPerspective(img_origin, M, (300,330))

    cv2.imshow('dst', dst)
    

### draw rectangle bbox
if args.with_draw == 'True':
    for bbox, confidence in zip(list_bboxes, list_confidence):
        l, t, r, b = bbox
        
        cv2.rectangle(img_origin, (l, t), (r, b),
            (0, 255, 0), 2)
        text = "face: %.2f" % confidence
        text_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = t #- 1 if t - 1 > 1 else t + 1
        cv2.rectangle(img_origin, (l,y-text_size[1]),(l+text_size[0], y+base_line), (0,255,0), -1)
        cv2.putText(img_origin, text, (l, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    for landmark in list_landmarks:
        for idx, point in enumerate(landmark):
            cv2.circle(img_origin, point, 2, (0, 255, 255), -1)
    for landmark in list_landmarks:
        cv2.circle(img_origin, landmark[48], 2, (0, 0, 255), -1)
        cv2.circle(img_origin, landmark[54], 2, (0, 0, 255), -1)
        cv2.circle(img_origin, landmark[30], 2, (0, 0, 255), -1)
        cv2.circle(img_origin, landmark[8], 2, (0, 0, 255), -1)
    
    cv2.imshow('show', img_origin)
    key = cv2.waitKey(0)
    if key == 27:
        exit()
