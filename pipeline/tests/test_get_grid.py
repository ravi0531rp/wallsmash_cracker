# from mss import mss
# import numpy as np
# import cv2
# import time
# import os
# from glob import glob
# from paddleocr import PaddleOCR
# from collections import Counter
# from tensorflow.keras.models import load_model
from ...pipeline.scripts import GetGridInfo
from loguru import logger

logger.info("Loading Model...")
# model = load_model("../digit_recog/models/model.h5")
# logger.success("Loaded")
# ocr = PaddleOCR(use_angle_cls=True, lang='en') 

# vals = get_grid(ocr, model)
# for v in vals:
#     logger.debug(v)