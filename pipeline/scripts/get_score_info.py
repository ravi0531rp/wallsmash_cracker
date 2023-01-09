from mss import mss
import numpy as np
import cv2

pt1 = (0, 212)
pt2 = (269,246)

monitor = {"top": pt1[1], "left": pt1[0], "width": (pt2[0]-pt1[0]), "height": (pt2[1]-pt1[1])}

def get_score(ocr):
    with mss() as sct:
        image = np.array(sct.grab(monitor))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        data = ocr.ocr(image)
        return int(np.array(data, dtype='object').squeeze()[1][1][0].replace(".",""))

if __name__ == "__main__":
    image = get_score("aa")


    