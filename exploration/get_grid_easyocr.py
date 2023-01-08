from mss import mss
import numpy as np
import cv2
import time
import os
from glob import glob
import easyocr
reader = easyocr.Reader(['en'])

from collections import Counter
from generate_references import generate_refs
from skimage.metrics import structural_similarity as ssim


thresh_images = generate_refs()
refer_dict_nums = {str(idx + 1) : thresh_images[idx] for idx in range(9)}
refer_dict_special = {f"Special_{str(idx - 8)}"  : thresh_images[idx] for idx in range(9,12)}



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


files = glob("../images/boxes8/*.jpg")
results = {}

counter = 0
for f in sorted(files):
    found = False
    result = reader.readtext(f)
    if result != [] and result != 0:
        result = result[0][1]
        results[f] = result
        found = True
    else:
        im1 = cv2.imread(f)
        gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)[50:100, 150:220]
        ret, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
        for k,v in refer_dict_nums.items():
            ssim_val = ssim(thresh, v)
            if ssim_val > 0.9:
                results[f] = k
                counter += 1
                break
        if not found:
            ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            for k,v in refer_dict_special.items():
                ssim_val = ssim(thresh, v)
                if ssim_val > 0.9:
                    results[f] = k
                    counter += 1
                    break
    if not found:
        results[f] = ""
        
            
for k,v in results.items():
    print(k,v)

print(counter)