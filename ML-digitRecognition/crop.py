import numpy as np
import cv2
import math


def find_contours(thresh):
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    centers = []
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

    return centers


def warp_transform(img, centers):
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

        max_x = int(np.ceil(max(math.sqrt(
            (new_bottom_right[0] - new_bottom_left[0]) ** 2 + (new_bottom_right[1] - new_bottom_left[1]) ** 2),
                                math.sqrt((new_top_left[0] - new_top_right[0]) ** 2 + (
                                            new_top_left[1] - new_top_right[1]) ** 2))))

        max_y = int(np.ceil(
            max(math.sqrt((new_top_left[0] - new_bottom_left[0]) ** 2 + (new_top_left[1] - new_bottom_left[1]) ** 2),
                math.sqrt(
                    (new_bottom_right[0] - new_top_right[0]) ** 2 + (new_top_right[1] - new_bottom_right[1]) ** 2))))

        points_A = np.float32([new_top_left, new_top_right, new_bottom_left, new_bottom_right])

        points_B = np.float32([[0, 0], [max_x, 0], [0, max_y], [max_x, max_y]])

        M = cv2.getPerspectiveTransform(points_A, points_B)

        return True, cv2.warpPerspective(img, M, (max_x, max_y))

    else:
        return False, np.zeros(img.shape, dtype='uint8')


def crop_digits(warped, save_dir):
    cropped_arr = []
    dig_count = 1
    for crop in range(0, 15):
        if crop == 0:
            temp = warped[2 + (crop * 145):2 + ((crop + 1) * 145), 800:1400]
        else:
            temp = warped[2 + (8 * crop) + (crop * 145):2 + (8 * crop) + (crop + 1) * 145, 800:1400]

        for dig in range(0, 5):
            if dig < 3:
                cropped_arr.append(temp[15:-15, ((10 * (dig + 1))) + (dig * 100):((dig + 1) * 100) + (10 * (dig + 1))])
                cv2.imwrite(save_dir + 'img' + str(dig_count) + '.jpg',
                    temp[15:-15, ((10 * (dig + 1))) + (dig * 100):((dig + 1) * 100) + (10 * (dig + 1))])
                dig_count = dig_count + 1
            else:
                cropped_arr.append(temp[15:-15:, 35 + ((10 * (dig + 1))) + (dig * 100):35 + ((dig + 1) * 100) + (10 * (dig + 1))])
                cv2.imwrite(save_dir + 'img' + str(dig_count) + '.jpg',
                    temp[15:-15:, 35 + ((10 * (dig + 1))) + (dig * 100):35 + ((dig + 1) * 100) + (10 * (dig + 1))])
                dig_count = dig_count + 1
    return cropped_arr


def turn_to_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 200, 255, 0)
    return thresh


def rotate_img(img, thresh):
    rotation_matrix = cv2.getRotationMatrix2D((thresh.shape[1] / 2, thresh.shape[0] / 2), 180, 1)
    second_thresh = cv2.warpAffine(thresh, rotation_matrix, (thresh.shape[1], thresh.shape[0]))

    rotation_matrix = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), 180, 1)
    second_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

    return second_img, second_thresh


def digitGen(img_path, save_dir):
    img = cv2.imread(img_path, 1)
    cropped_arr = []

    thresh = turn_to_binary(img)
    centers = find_contours(thresh)

    if len(centers) != 30:
        img, thresh = rotate_img(img, thresh)
        centers = find_contours(thresh)

    if len(centers) == 30:
        flag, warped = warp_transform(img, centers)
        if (flag == False):
            print('Zero denominator for warped image')
        else:
            cropped_arr = crop_digits(warped, save_dir)
    else:
        print("Number of contours is not equivalent to 30")


digitGen("/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/DigitDetection/Forms/form1/00000001.jpg",
         "/home/bardia/Documents/University/Seventh Semester/Principles of Computational Intelligence/DigitDetection/temp/")
