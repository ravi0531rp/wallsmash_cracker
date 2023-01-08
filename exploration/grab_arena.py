from mss import mss
import numpy as np
import cv2
import time
import os
from glob import glob
from paddleocr import PaddleOCR,draw_ocr
from collections import Counter

ocr = PaddleOCR(use_angle_cls=True, lang='en') 

time.sleep(3)
pt1 = (0, 213)
pt2 = (1790,1038)

monitor = {"top": pt1[1], "left": pt1[0], "width": (pt2[0]-pt1[0]), "height": (pt2[1]-pt1[1])}

with mss() as sct:
    image = np.array(sct.grab(monitor))
    print(image.shape)
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


files = glob("/Users/raprak-blrm20/Desktop/personal/CODES/m-projects/a-games/wallsmash_cracker/images/boxes8/*.jpg")
ocr = PaddleOCR(use_angle_cls=True, lang='en') 
results = {}
data = {}
for f in files:
    result = ocr.ocr(f, cls=True)
    if result != [[]]:
        # data[f] = np.array(result).squeeze()[1]
        result = np.array(result).squeeze()[1][0]
        results[f] = result


vals = list(results.values())
print(set(vals))

print(Counter(vals))
# print(data)
