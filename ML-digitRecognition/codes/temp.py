import glob
import numpy as np
import cv2
import math

not_cropped = []

base = "/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/DigitDetection/"
for form in range(1, 2):
    num_imgs = len(glob.glob(base + "Forms/form" + str(form) + '/' + "*.jpg"))
    for IMG in range(23, 24):
    #def main():
        general_path = base + "Forms/form" + str(form) + "/000000"
        if IMG < 10:
            imgPath = general_path + str(0) + str(IMG) + '.jpg' 
        else:
            imgPath = general_path + str(IMG) + '.jpg'

        centers = []
        img = cv2.imread(imgPath, 1)

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 200, 255, 0)

        kernel = np.ones((5,5), np.uint8)


        img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

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
                    

                    
                    
        '''if len(centers) != 30:
            rotation_matrix = cv2.getRotationMatrix2D((thresh.shape[1]/2, thresh.shape[0]/2), 180, 1)
            thresh = cv2.warpAffine(thresh, rotation_matrix, (thresh.shape[1], thresh.shape[0]))
            
            rotation_matrix = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), 180, 1)
            img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
            
            contoursList = []
            counter = 0
            centers = []
            
            img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


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
                        centers.append([cx, cy])'''
        
        cv2.drawContours(img, contoursList, -1, (255, 0, 0), 8)
        original = cv2.imread(imgPath, 1)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        #cv2.imwrite('ss.jpg', img)
        print(contoursList[0][0])
        output = [original, img]
        titles = ['Original', 'Contours']
        #print("counter: ", counter)
        #print('counter list: ', contoursList[0])
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.imshow(output[i])
            plt.title(titles[i])
            plt.xticks([])
            plt.yticks([])

            plt.show()

        if len(centers) == 30:
            if centers[29][0] - centers[1][0] != 0 and centers[28][0] - centers[0][0] != 0:
                left_deg = (centers[29][1] - centers[1][1]) / (centers[29][0] - centers[1][0])
                right_deg = (centers[28][1] - centers[0][1]) / (centers[28][0] - centers[0][0])

                ratio = 80

                avr_deg = (left_deg + right_deg) / 2
    
    
                new_top_left = [int(np.floor(centers[29][0] + abs(math.cos(math.atan(avr_deg)) * ratio))), 
                                int(np.floor(centers[29][1] - abs(math.sin(math.atan(avr_deg)) * ratio)))]

                new_top_right = [int(np.ceil(centers[28][0] + abs(math.cos(math.atan(avr_deg)) * ratio))), 
                                 int(np.floor(centers[28][1] - abs(math.sin(math.atan(avr_deg)) * ratio)))]

                new_bottom_left = [int(np.floor(centers[1][0] - abs(math.cos(math.atan(avr_deg)) * ratio))), 
                                   int(np.ceil(centers[1][1] + abs(math.sin(math.atan(avr_deg)) * ratio)))]

                new_bottom_right = [int(np.ceil(centers[0][0] - abs(math.cos(math.atan(avr_deg)) * ratio))), 
                                    int(np.ceil(centers[0][1] + abs(math.sin(math.atan(avr_deg)) * ratio)))]

                #print(abs(math.cos(math.atan(right_deg)) * ratio))

                max_x = int(np.ceil(max(math.sqrt((new_bottom_right[0] - new_bottom_left[0]) ** 2 + (new_bottom_right[1] - new_bottom_left[1]) ** 2),
                                   math.sqrt((new_top_left[0] - new_top_right[0]) ** 2 + (new_top_left[1] - new_top_right[1]) ** 2))))

                max_y = int(np.ceil(max(math.sqrt((new_top_left[0] - new_bottom_left[0]) ** 2 + (new_top_left[1] - new_bottom_left[1]) ** 2), 
                                   math.sqrt((new_bottom_right[0] - new_top_right[0]) ** 2 + (new_top_right[1] - new_bottom_right[1]) ** 2))))

                points_A = np.float32([new_top_left, new_top_right, new_bottom_left, new_bottom_right])

                points_B = np.float32([[0,0], [max_x,0], [0,max_y], [max_x,max_y]])

                M = cv2.getPerspectiveTransform(points_A, points_B)

                warped = cv2.warpPerspective(img, M, (max_x, max_y))
                
                cv2.imwrite('23.jpg', warped)

                '''for crop in range (0, 15):
                    #cv2.imshow('d', warped[(crop * 150): ((crop + 1) * 150), 30:100])
                    #cv2.imwrite(base + 'temp/dfs'+str(crop) + '.jpg', warped[crop * 20 + (crop * 130): crop * 20 + ((crop + 1) * 130), 800:1400])

                    if crop == 0:
                        temp =  warped[2 + (crop * 145):2 + ((crop + 1) * 145), 800:1400]
                        #cv2.imwrite(base + 'temp/crop/IMG' + str(IMG) + '_row' + str(crop) + '.jpg', temp)
            
                    else:
                        temp = warped[2 + (8 * crop) + (crop * 145):2 + (8 * crop) + (crop + 1) * 145, 800:1400]
                        #cv2.imwrite(base + 'temp/crop/IMG' + str(IMG) + '_row' + str(crop) + '.jpg', temp)

                        #cv2.imwrite(base + 'temp/crop/dig' + str(crop) + '.jpg', temp)
                    for dig in range (0, 5):
                        if dig < 3:
                            cv2.imwrite(base + 'temp/crop/Form' + str(form) + '/IMG' + str(IMG) + '_row' + str(crop) + '_dig' + str(dig) + '.jpg', temp[15:-15, ((10 * (dig + 1))) + (dig * 100):((dig + 1) * 100) + (10 * (dig + 1))])
                        else:
                            cv2.imwrite(base + 'temp/crop/Form' + str(form) + '/IMG' + str(IMG) + '_row' + str(crop) + '_dig' + str(dig) + '.jpg', temp[15:-15:, 35 + ((10 * (dig + 1))) + (dig * 100):35 + ((dig + 1) * 100) + (10 * (dig + 1))])'''

            else:
                print("in form" + str(form) + " image" + str(IMG) + " has" + " zero Denominator")
        else:
            print("in form" + str(form) + " image" + str(IMG) + " has " + str(len(centers)) + " centers the first time")