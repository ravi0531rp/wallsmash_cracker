from mss import mss
import numpy as np
import cv2

pt1 = (817, 685)
pt2 = (989,733)

monitor = {"top": pt1[1], "left": pt1[0], "width": (pt2[0]-pt1[0]), "height": (pt2[1]-pt1[1])}

def check_end(ocr):
    with mss() as sct:
        image = np.array(sct.grab(monitor))
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        data = ocr.ocr(image)
        res = np.array(data, dtype='object').squeeze()
        try:
            return res[1][0] == "[tap to restart]"
        except:
            return False


if __name__ == "__main__":
    image = check_end("aa")


    