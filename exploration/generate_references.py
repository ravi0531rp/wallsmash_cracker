import numpy as np
import cv2

def generate_refs():
    im_1 = cv2.imread("/Users/raprak-blrm20/Desktop/personal/CODES/m-projects/a-games/wallsmash_cracker/images/references/1.jpg")
    gray = cv2.cvtColor(im_1, cv2.COLOR_BGR2GRAY)[50:100, 150:220]
    _, thresh1 = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    im_2 = cv2.imread("/Users/raprak-blrm20/Desktop/personal/CODES/m-projects/a-games/wallsmash_cracker/images/references/2.jpg")
    gray = cv2.cvtColor(im_2, cv2.COLOR_BGR2GRAY)[50:100, 150:220]
    _, thresh2 = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    im_3 = cv2.imread("/Users/raprak-blrm20/Desktop/personal/CODES/m-projects/a-games/wallsmash_cracker/images/references/3.jpg")
    gray = cv2.cvtColor(im_3, cv2.COLOR_BGR2GRAY)[50:100, 150:220]
    _, thresh3 = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    im_4 = cv2.imread("/Users/raprak-blrm20/Desktop/personal/CODES/m-projects/a-games/wallsmash_cracker/images/references/4.jpg")
    gray = cv2.cvtColor(im_4, cv2.COLOR_BGR2GRAY)[50:100, 150:220]
    _, thresh4 = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    im_5 = cv2.imread("/Users/raprak-blrm20/Desktop/personal/CODES/m-projects/a-games/wallsmash_cracker/images/references/5.jpg")
    gray = cv2.cvtColor(im_5, cv2.COLOR_BGR2GRAY)[50:100, 150:220]
    _, thresh5 = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    im_6 = cv2.imread("/Users/raprak-blrm20/Desktop/personal/CODES/m-projects/a-games/wallsmash_cracker/images/references/6.jpg")
    gray = cv2.cvtColor(im_6, cv2.COLOR_BGR2GRAY)[50:100, 150:220]
    _, thresh6 = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    im_7 = cv2.imread("/Users/raprak-blrm20/Desktop/personal/CODES/m-projects/a-games/wallsmash_cracker/images/references/7.jpg")
    gray = cv2.cvtColor(im_7, cv2.COLOR_BGR2GRAY)[50:100, 150:220]
    _, thresh7 = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    im_8 = cv2.imread("/Users/raprak-blrm20/Desktop/personal/CODES/m-projects/a-games/wallsmash_cracker/images/references/8.jpg")
    gray = cv2.cvtColor(im_8, cv2.COLOR_BGR2GRAY)[50:100, 150:220]
    _, thresh8 = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    im_9 = cv2.imread("/Users/raprak-blrm20/Desktop/personal/CODES/m-projects/a-games/wallsmash_cracker/images/references/9.jpg")
    gray = cv2.cvtColor(im_9, cv2.COLOR_BGR2GRAY)[50:100, 150:220]
    _, thresh9 = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    im_black = cv2.imread("/Users/raprak-blrm20/Desktop/personal/CODES/m-projects/a-games/wallsmash_cracker/images/references/black.jpg")
    gray = cv2.cvtColor(im_black, cv2.COLOR_BGR2GRAY)[50:100, 150:220]
    _, thresh_black = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    im_ball = cv2.imread("/Users/raprak-blrm20/Desktop/personal/CODES/m-projects/a-games/wallsmash_cracker/images/references/ball.jpg")
    gray = cv2.cvtColor(im_ball, cv2.COLOR_BGR2GRAY)[50:100, 150:220]
    _, thresh_ball = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    im_position = cv2.imread("/Users/raprak-blrm20/Desktop/personal/CODES/m-projects/a-games/wallsmash_cracker/images/references/position.jpg")
    gray = cv2.cvtColor(im_position, cv2.COLOR_BGR2GRAY)[50:100, 150:220]
    _, thresh_position = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    return [thresh1, thresh2, thresh3, thresh4, thresh5, thresh6, thresh7, thresh8, thresh9, thresh_black, thresh_ball, thresh_position]


if __name__ == "__main__":
    thresh_images = generate_refs()
    for th in thresh_images:
        print(th.shape)