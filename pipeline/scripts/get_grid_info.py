from mss import mss
import numpy as np
import cv2
import time
import os
from loguru import logger

pt1 = (0, 213)
pt2 = (1790,1038)

monitor = {"top": pt1[1], "left": pt1[0], "width": (pt2[0]-pt1[0]), "height": (pt2[1]-pt1[1])}


def get_grid(ocr, model):

    try:
        with mss() as sct:
            image = np.array(sct.grab(monitor))
            arena = image[60:1600,:]

        nRows = 10
        mCols = 10
        sizeX = arena.shape[1]
        sizeY = arena.shape[0]

        rois = []
        for i in range(0,nRows):
            for j in range(0, mCols):
                roi = arena[i*sizeY//nRows:i*sizeY//nRows + sizeY//nRows ,j*sizeX//mCols:j*sizeX//mCols + sizeX//mCols]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
                rois.append(roi)
        results = []

        for idx, roi in enumerate(rois):
            im = roi[50:100, 150:220]
            im = cv2.resize(im, (100,100))

            result = ocr.ocr(roi, cls=True)
            done = False

            if result != [[]]:
                res = np.array(result , dtype='object').squeeze()[1][0]
                if res.isnumeric():
                    if int(res) > 9:
                        results.append(int(res))
                        done = True

            if not done:
                image = im.reshape(1,100,100,3)
                pred = np.argmax(model.predict(image))
                if pred < 9:
                    results.append(pred + 1)
                else:
                    if pred == 9:
                        results.append("Ball")
                    elif pred ==10:
                        results.append("Blank")
                    elif pred == 11:
                        results.append("marker")
                    
        board_state = [[results[i] for i in range(j, j+10)] for j in range(0,100,10)]
        return board_state 
    except Exception as e:
        logger.error(e)
        return []



    