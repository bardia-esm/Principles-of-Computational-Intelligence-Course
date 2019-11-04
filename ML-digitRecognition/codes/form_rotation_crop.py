import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os, shutil

not_cropped = []

for IMG in range(1, 2):
#def main():
    general_path = "Forms/form1/000000"
    if IMG < 10:
        imgPath = general_path + str(0) + str(IMG) + '.jpg' 
    else:
        imgPath = general_path + str(IMG) + '.jpg'
    
    centers = []
    
    img = cv2.imread(imgPath, 1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 200, 255, 0)
    
    kernel = np.ones((5,5), np.uint8)


    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #print("contour shape: ", np.shape(contours))
    contoursList = []
    counter = 0

    for i in range(np.shape(contours)[0]):
        cnt = contours[i]
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        if (cx >= 2218 or cx <= 240) and (680 <= cy <= 2946):
            if 3200.0 <= cv2.contourArea(contours[i]) <= 3800.0 or 248.0 <= cv2.arcLength(contours[i], True) <= 265.0:
                contoursList.append(contours[i])
                counter += 1
                centers.append([cx, cy])
    
    
    if centers[29][0] - centers[1][0] != 0 and centers[28][0] - centers[0][0] != 0:
        left_deg = (centers[29][1] - centers[1][1]) / (centers[29][0] - centers[1][0])
        right_deg = (centers[28][1] - centers[0][1]) / (centers[28][0] - centers[0][0])

        ratio = 60

        new_top_left = [int(np.floor(centers[29][0] + abs(math.cos(math.atan(left_deg)) * ratio))), 
                        int(np.floor(centers[29][1] - abs(math.sin(math.atan(left_deg)) * ratio)))]

        new_top_right = [int(np.ceil(centers[28][0] + abs(math.cos(math.atan(right_deg)) * ratio))), 
                         int(np.floor(centers[28][1] - abs(math.sin(math.atan(right_deg)) * ratio)))]

        new_bottom_left = [int(np.floor(centers[1][0] - abs(math.cos(math.atan(left_deg)) * ratio))), 
                           int(np.ceil(centers[1][1] + abs(math.sin(math.atan(left_deg)) * ratio)))]

        new_bottom_right = [int(np.ceil(centers[0][0] - abs(math.cos(math.atan(right_deg)) * ratio))), 
                            int(np.ceil(centers[0][1] + abs(math.sin(math.atan(right_deg)) * ratio)))]

        print(abs(math.cos(math.atan(right_deg)) * ratio))

        max_x = int(np.ceil(max(math.sqrt((new_bottom_right[0] - new_bottom_left[0]) ** 2 + (new_bottom_right[1] - new_bottom_left[1]) ** 2),
                           math.sqrt((new_top_left[0] - new_top_right[0]) ** 2 + (new_top_left[1] - new_top_right[1]) ** 2))))

        max_y = int(np.ceil(max(math.sqrt((new_top_left[0] - new_bottom_left[0]) ** 2 + (new_top_left[1] - new_bottom_left[1]) ** 2), 
                           math.sqrt((new_bottom_right[0] - new_top_right[0]) ** 2 + (new_top_right[1] - new_bottom_right[1]) ** 2))))

        points_A = np.float32([new_top_left, new_top_right, new_bottom_left, new_bottom_right])

        points_B = np.float32([[0,0], [max_x,0], [0,max_y], [max_x,max_y]])

        M = cv2.getPerspectiveTransform(points_A, points_B)

        warped = cv2.warpPerspective(thresh, M, (max_x, max_y))
        
    
    
    
    padded = add_padding(warped, 80)
    
    cv2.imwrite('temp/wraped_padded' + str (IMG) + '.jpg', padded)
    cv2.imwrite('temp/wraped' + str (IMG) + '.jpg', warped)
    
    
    
    contoursList = []
    centers = []
    counter = 0
    
    ret, thresh_padded = cv2.threshold(padded, 200, 255, 0)

    img2, contours, hierarchy = cv2.findContours(thresh_padded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for i in range(np.shape(contours)[0]):
        cnt = contours[i]
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        '''if (cx >= 1190 or cx <= 92) and (68 <= cy <= 1320):
            if 570.0 <= cv2.contourArea(contours[i]) <= 663.0 or 97.0 <= cv2.arcLength(contours[i], True) <= 104.0:'''
        if (cx >= 2115 or cx <= 91) and (46 <= cy <= 2295):
            if 1773.0 <= cv2.contourArea(contours[i]) <= 2045.0 or 170.0 <= cv2.arcLength(contours[i], True) <= 190.0:
                contoursList.append(contours[i])
                counter += 1
                centers.append([cx, cy])
                
                
    print(len(centers))
    
    for crp in range(0, 29, 2):
        #print(centers[crp], centers[crp + 1])
        cropped = gray[centers[crp][1] - 30:centers[crp][1] + 30, 477:805]
        cv2.imwrite('temp/z' + str(crp) + '.jpg', gray[centers[crp][1] - 60:centers[crp][1] + 60, 850:1600])

        '''for dig in range(0, 5):
            if dig < 3:
                cv2.imwrite('temp/img' + str(IMG) + '_croped' + str(crp) + '_dig' + str(dig) + '.jpg', cropped[:, (56 * dig) + (dig * 5): (56 * (dig + 1)) + (dig * 5)])
            else:
                cv2.imwrite('temp/croped' + str(crp) + '_dig' + str(dig) + '.jpg', cropped[:, (56 * dig) + 21 + (dig * 5): (56 * (dig + 1)) + 21 + (dig * 5)])
    '''
    cv2.waitKey(0)
    cv2.destroyAllWindows()