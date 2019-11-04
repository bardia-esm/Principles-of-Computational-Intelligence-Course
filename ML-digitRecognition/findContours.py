import cv2
import matplotlib.pyplot as plt
import numpy as np


def readImg(imgPath):
    img = cv2.imread(imgPath, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def createGray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def thresh(grayImg):
    ret, thresh = cv2.threshold(grayImg, 200, 255, 0)
    return thresh


def contours(thresh):
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


def filterContours(contours):
    contoursList = []
    counter = 0
    centers = []

    for i in range(np.shape(contours)[0]):
        cnt = contours[i]
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        if (cx >= 2200 or cx <= 240) and (680 <= cy <= 3072):
            if 3000.0 <= cv2.contourArea(contours[i]) <= 3820.0 or 248.0 <= cv2.arcLength(contours[i], True) <= 270.0:
                contoursList.append(contours[i])
                counter += 1
                centers.append([cx, cy])
    return contoursList, centers


def drawContours(img, contoursList):
    cv2.drawContours(img, contoursList, -1, (255, 0, 0), 8)
    return img


def plotContours(originalImg, contoursImg):
    output = [originalImg, contoursImg]
    titles = ['Original', 'Contours']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(output[i])
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])

    plt.show()


# read image
# path = "E:/Users/asus/Desktop/Desktop Folders/7th semester/ML/Forms/form1/"
# img = readImg(path, '00000001.jpg')

# gray scale image
# gray = createGray(img)

# get the threshold
# thresh = thresh(gray)

# get the contours
# contours = contours(thresh)

# find the eyes
# contoursList, centers = filterContours(contours)

# contours image
# contoursImg = drawContours(img, contoursList)

# plot contours
# plotContours(gray, contoursImg)


# print("contour shape: ", np.shape(contours))

# else:
#     print("contourArea: ", cv2.contourArea(contours[i]))
#     print("arcLength: ", cv2.arcLength(contours[i], True))

# if 3000.0 <= cv2.contourArea(contours[i]) <= 3820.0 or 248.0 <= cv2.arcLength(contours[i], True) <= 270.0:
#     if not (cx >= 2218 or cx <= 240):
#         print('cx: ', cx, 'cy: ', cy)
#         print("contourArea: ", cv2.contourArea(contours[i]))
#         print("arcLength: ", cv2.arcLength(contours[i], True))

#
# print("cx, cy: ", cx, cy)
# print("contourArea: ", cv2.contourArea(cnt))
# print("arcLength: ", cv2.arcLength(cnt, True))

# x, y, w, h = cv2.boundingRect(cnt)
# cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# original = cv2.imread(imgPath, 1)
# original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)


# degree = atan((centers[29][1] - centers[28][1]) / (centers[29][0] - centers[28][0]))
# rotation_matrix = cv2.getRotationMatrix2D((0, 0), degree, 0.5)
# rotated_image = cv2.warpAffine(gray, rotation_matrix, (gray.shape[0], gray.shape[1]))

# print(degree)
# print(centers)
# cv2.imwrite('rotated.jpg', rotated_image)
# cv2.imwrite('original.jpg', thresh)
# print(thresh.shape)
# print(rotated_image.shape)
