import easyocr
import os
from glob import glob
import cv2
reader = easyocr.Reader(['en'])

files = glob("/Users/raprak-blrm20/Desktop/personal/CODES/m-projects/a-games/wallsmash_cracker/images/boxes/*.jpg")

for f in files[:10]:
    im = cv2.imread(f)
    result = reader.readtext(f)
    text = "NA"
    if result != []:
        text = result[0][1]
    cv2.imshow(text, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()