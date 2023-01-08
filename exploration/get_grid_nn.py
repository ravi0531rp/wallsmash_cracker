from mss import mss
import numpy as np
import cv2
import time
import os
from glob import glob
from paddleocr import PaddleOCR
from collections import Counter
from generate_references import generate_refs
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model

ocr = PaddleOCR(use_angle_cls=True, lang='en') 

time.sleep(3)
pt1 = (0, 213)
pt2 = (1790,1038)

monitor = {"top": pt1[1], "left": pt1[0], "width": (pt2[0]-pt1[0]), "height": (pt2[1]-pt1[1])}

with mss() as sct:
    image = np.array(sct.grab(monitor))
    arena = image[60:1600,:]


nRows = 10
mCols = 10
sizeX = arena.shape[1]
sizeY = arena.shape[0]
save_path = "../images/boxes8/"
try:
    os.makedirs(save_path)
except:
    pass

for i in range(0,nRows):
    for j in range(0, mCols):
        roi = arena[i*sizeY//nRows:i*sizeY//nRows + sizeY//nRows ,j*sizeX//mCols:j*sizeX//mCols + sizeX//mCols]
        cv2.imwrite(f'{save_path}patch_'+str(i)+str(j)+".jpg", roi)

model = load_model("../digit_recog/models/model.h5")

files = glob("../images/boxes8/*.jpg")
files = sorted(files)
ocr = PaddleOCR(use_angle_cls=True, lang='en') 
results = []

for idx, f in enumerate(files):
    im = cv2.imread(f)[50:100, 150:220]
    im = cv2.resize(im, (100,100))

    result = ocr.ocr(f, cls=True)
    done = False

    if result != [[]]:
        res = np.array(result).squeeze()[1][0]
        if res.isnumeric():
            if int(res) > 9:
                results.append(res)
                done = True

    if not done:
        image = im.reshape(1,100,100,3)
        pred = np.argmax(model.predict(image))
        if pred < 9:
            results.append(str(pred + 1))
        else:
            if pred == 9:
                results.append("Ball")
            elif pred ==10:
                results.append("Blank")
            elif pred == 11:
                results.append("marker")
            


        
vals = [[results[i] for i in range(j, j+10)] for j in range(0,100,10)]

for v in vals:
    print(v)