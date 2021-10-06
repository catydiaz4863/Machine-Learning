import cv2
import os


files = os.listdir("/Users/catalinadiaz/Desktop/b/A-jpg/")
for i in range(len(files)):
    print(files[i])
    img = cv2.imread("/Users/catalinadiaz/Desktop/b/A-jpg/"+files[i], cv2.IMREAD_UNCHANGED)
    width = 500
    height = 500
    dim = (width, height)
    # # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    status = cv2.imwrite('/Users/catalinadiaz/Documents/temp/' + str(i+180) + '.jpg', resized)
