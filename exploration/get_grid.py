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


ocr = PaddleOCR(use_angle_cls=True, lang='en') 

thresh_images = generate_refs()
refer_dict_nums = {str(idx + 1) : thresh_images[idx] for idx in range(9)}
refer_dict_special = {"black" : thresh_images[9] , "ball" : thresh_images[10], "position" : thresh_images[11]}


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


files = glob("../images/boxes8/*.jpg")
ocr = PaddleOCR(use_angle_cls=True, lang='en') 
results = {}
counter = 0

for f in sorted(files):
    im1 = cv2.imread(f)
    result = ocr.ocr(f, cls=True)
    if result != [[]] and result != 0:
        result = np.array(result).squeeze()[1][0]
        results[f] = result
    else:
        gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)[50:100, 150:220]
        ret, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
        ssim_vals_1 = {}
        for k,v in refer_dict_nums.items():
            ssim_val = ssim(thresh, v)
            ssim_vals_1[k] = ssim_val

        print(ssim_vals_1)
        sorted_vals  = sorted(ssim_vals_1.items(), key = lambda x: x[1], reverse=True)
        print(ssim_vals_1)
        best_match = sorted_vals[0][0]
        best_val = sorted_vals[0][1]
        

        ssim_vals_2 = {}
        ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        for k,v in refer_dict_special.items():
            ssim_val = ssim(thresh, v)
            ssim_vals_2[k] = ssim_val

        sorted_vals  = sorted(ssim_vals_2.items(), key = lambda x: x[1], reverse=True)
        best_match_2 = sorted_vals[0][0]
        best_val_2 = sorted_vals[0][1]
        print(best_match)
        print(best_match_2)
    
        if best_val_2 > best_val:
            results[f] = best_match_2
        else:
            results[f] = best_match
    cv2.imshow(results[f], im1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


values = list(results.values())
        
vals = [[values[i] for i in range(j, j+10)] for j in range(0,100,10)]



for v in vals:
    print(v)