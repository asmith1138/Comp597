import numpy as np
import matplotlib.pyplot as plt
import cv2


image = cv2.imread('rose.jpg')
plt.imshow(image)

sobel_x = np.array([[ -1, 0, 1], 
                   [ -2, 0, 2], 
                   [ -1, 0, 1]]) 
				   
gaussian = 1/9 * np.array([[ 1, 1, 1], 
					 	[ 1, 1, 1], 
						[ 1, 1, 1]]) 	

''' convolution operation '''
filtered_image = cv2.filter2D(image, -1, sobel_x)						

plt.imshow(filtered_image)
