import numpy as np
import cv2

#opencv-python dan opencv contrib python 3.4.2.17
image1 = cv2.imread("aldi1.jpg")
image2= cv2.imread('aldi2.jpg')
#1)check if 2 images are equals
if image1.shape == image2.shape:
	print('the images have same size and channels')
	difference = cv2.subtract (image1, image2)
	r,g,b = cv2.split(difference)
	if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r)==0:
		print('the images are completely equal')
	else:
		print('the images are not equal')

#2) check error
def mse(image1, image2):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image2.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

z = mse(image1, image2)
print('nilai error adalah',z)

# 3) check for similaritie between the 2 images
sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(image1, None)
kp_2, desc_2 = sift.detectAndCompute(image2, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann =  cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch (desc_1, desc_2, k=2)

result = cv2.drawMatchesKnn(image1, kp_1, image2, kp_2, matches, None)

original_img = cv2.hconcat((image1, image2))
cv2.imwrite('resultsimilarity.jpg',result)
cv2.imwrite('difference.jpg',difference)
cv2.imshow('original',cv2.resize(original_img,None, fx=0.4, fy=0.4))
cv2.imshow('difference', cv2.resize(difference,None, fx=0.4, fy=0.4))
cv2.imshow('result', cv2.resize(result,None, fx=0.4, fy=0.4))
cv2.waitKey(0)
cv2.destroyAllWindows()
