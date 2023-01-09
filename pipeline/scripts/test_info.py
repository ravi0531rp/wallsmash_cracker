from mss import mss
import numpy as np
import cv2
import time
import os
from glob import glob
from utils.sleep import stay_idle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from paddleocr import PaddleOCR
from collections import Counter
from tensorflow.keras.models import load_model
from loguru import logger
from get_grid_info import get_grid
from get_score_info import get_score
from get_endgame_info import check_end

logger.info("Loading Model...")

model = load_model("/Users/raprak-blrm20/Desktop/personal/CODES/m-projects/a-games/wallsmash_cracker/digit_recog/models/model.h5")
logger.success("Loaded")
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False) 

stay_idle(2)

end = check_end(ocr)

if not end:
    logger.info("Game Still On")
    board_state = get_grid(ocr, model)
    logger.debug('\n'.join(['\n'] + ['\t'.join([str(cell) for cell in row]) for row in board_state]))

score = get_score(ocr)
print(f"Current Score is {score}")

