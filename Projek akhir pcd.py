import cv2
import numpy as np

#detection
bgr = cv2.imread('citra/test-case-3.2.jpg')

hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
lower1 = np.array([160, 25, 25], dtype=np.uint8)
upper1 = np.array([180, 255, 255], dtype=np.uint8)
mask1 = cv2.inRange(hsv, lower1, upper1)
lower2 = np.array([0, 200, 100], dtype=np.uint8)
upper2 = np.array([30, 255, 255], dtype=np.uint8)
mask2 = cv2.inRange(hsv, lower2, upper2)
mask = mask1 + mask2
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#filling the hole
im_floodfill = mask.copy()
h, w = mask.shape[:2] # Notice the size needs to be 2 pixels than the image.
maskfill = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(im_floodfill, maskfill, (0,0), 255) # Floodfill from point (0, 0)
im_floodfill_inv = 255 - im_floodfill
circle = mask+im_floodfill_inv
masked2 = cv2.bitwise_and(bgr, bgr, mask=circle)   #masked traffic sign area

#crop img
points = np.argwhere(circle==255) # find where the white pixels are
points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
x, y, w, h = cv2.boundingRect(points) # create a rectangle around those points
crop = bgr[y:y+h, x:x+w] # create a cropped region for display
crop2 = masked2[y:y+h, x:x+w] # create a cropped region for feature matching
cv2.rectangle(bgr, (x, y), (x+w, y+h), (0, 255, 255), 3)

#resize img
rambu = cv2.resize(crop2, (400, 400))
rambu_gray = cv2.cvtColor(rambu, cv2.COLOR_BGR2GRAY)
for y in range(400):
    for x in range(400):
        new_value = rambu_gray[x][y]
        new_value = 255 if new_value == 0 else new_value
        rambu_gray[x][y] = new_value

#optimizing color
for y in range(400): #change contrast
    for x in range(400):
        new_value = (rambu_gray[x][y])*1.2
        new_value = 255 if new_value > 255 else new_value
        rambu_gray[x][y] = new_value
rambu_gray_copy = rambu_gray.copy()
retval, rambu_biner = cv2.threshold(rambu_gray_copy, thresh=175, maxval=255, type=cv2.THRESH_BINARY)
rambu_inv = 255 - rambu_biner

#template matching
#case 1, dilarang belok kiri
template1 = cv2.imread('citra/template-1.jpg')
template_gray1 = cv2.cvtColor(template1, cv2.COLOR_BGR2GRAY)
retval, template_biner1 = cv2.threshold(template_gray1, 0, 255, cv2.THRESH_OTSU)
template_biner1 = 255 - template_biner1
intersec1 = cv2.bitwise_and(rambu_inv, template_biner1)
counterI1 = 0
for y in range(400):
    for x in range(400):
        px = intersec1[x][y]
        if px == 255:
            counterI1 += 1
counterT1 = 0
for y in range(400):
    for x in range(400):
        px = template_biner1[x][y]
        if px == 255:
            counterT1 += 1
hasil1 = round(float(counterI1)/counterT1, 5)
print 'counterI1:', counterI1
print 'counterT1:', counterT1
print 'Akurasi case 1: ', hasil1

#case 2, dilarang masuk
template2 = cv2.imread('citra/template-2.jpg')
template_gray2 = cv2.cvtColor(template2, cv2.COLOR_BGR2GRAY)
retval, template_biner2 = cv2.threshold(template_gray2, 0, 255, cv2.THRESH_OTSU)
template_biner2 = 255 - template_biner2
intersec2 = cv2.bitwise_and(rambu_inv, template_biner2)
counterI2 = 0
for y in range(400):
    for x in range(400):
        px = intersec2[x][y]
        if px == 255:
            counterI2 += 1
counterT2 = 0
for y in range(400):
    for x in range(400):
        px = template_biner2[x][y]
        if px == 255:
            counterT2 += 1
hasil2 = round(float(counterI2)/counterT2, 5)
print 'counterI2:', counterI2
print 'counterT2:', counterT2
print 'Akurasi case 2: ', hasil2

#case 2, dilarang parkir
template3 = cv2.imread('citra/template-3.jpg')
template_gray3 = cv2.cvtColor(template3, cv2.COLOR_BGR2GRAY)
retval, template_biner3 = cv2.threshold(template_gray3, 0, 255, cv2.THRESH_OTSU)
template_biner3 = 255 - template_biner3
intersec3 = cv2.bitwise_and(rambu_inv, template_biner3)
counterI3 = 0
for y in range(400):
    for x in range(400):
        px = intersec3[x][y]
        if px == 255:
            counterI3 += 1
counterT3 = 0
for y in range(400):
    for x in range(400):
        px = template_biner3[x][y]
        if px == 255:
            counterT3 += 1
hasil3 = round(float(counterI3)/counterT3, 5)
print 'counterI3:', counterI3
print 'counterT3:', counterT3
print 'Akurasi case 3: ', hasil3

#menentukan rambu
font = cv2.FONT_ITALIC
if (hasil1 > hasil2) & (hasil1 > hasil3):
    cv2.putText(bgr, 'Dilarang belok kiri', (x-w, y), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA);
elif (hasil1 > hasil2) & (hasil1 < hasil3):
    cv2.putText(bgr, 'Dilarang parkir', (x-w, y), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA);
elif (hasil2 > hasil3) & (hasil2 > hasil1):
    cv2.putText(bgr, 'Dilarang masuk', (x-w, y), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA);
elif (hasil2 > hasil3) & (hasil2 < hasil1):
    cv2.putText(bgr, 'Dilarang belok kiri', (x-w, y), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA);
elif (hasil3 > hasil1) & (hasil3 > hasil2):
    cv2.putText(bgr, 'Dilarang parkir', (x-w, y), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA);
elif (hasil3 > hasil1) & (hasil3 < hasil2):
    cv2.putText(bgr, 'Dilarang masuk', (x-w, y), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA);

# cv2.imshow('brg', bgr)
# cv2.imshow('crop2', crop2)
# cv2.imshow('masked2', masked2)
# cv2.imshow('rambu', rambu)

cv2.imwrite('citra/test-case-3.2-rambu.jpg', rambu)
cv2.imwrite('citra/test-case-3.2-detection.jpg', bgr)


# cv2.imshow('rambu biner', rambu_inv)
# cv2.imshow('template biner1', template_biner1)
# cv2.imshow('hasil 1', intersec1)
# cv2.imshow('template biner2', template_biner2)
# cv2.imshow('hasil 2', intersec2)
# cv2.imshow('template biner3', template_biner3)
# cv2.imshow('hasil 3', intersec3)
cv2.waitKey(0)
cv2.destroyAllWindows()