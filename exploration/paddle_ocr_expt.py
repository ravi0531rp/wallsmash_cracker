from paddleocr import PaddleOCR,draw_ocr

ocr = PaddleOCR(use_angle_cls=True, lang='en') 
img_path = "../images/boxes/patch_22.jpg"
result = ocr.ocr(img_path, cls=True)
print(result)
# for idx in range(len(result)):
#     res = result[idx]
#     for line in res:
#         print(line)