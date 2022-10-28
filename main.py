import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread('shutterstock.jpg')

# Make float and divide by 255 to give BGRdash
bgrdash = img.astype(np.float64)/255.

# Calculate K as (1 - whatever is biggest out of Rdash, Gdash, Bdash)
K = 1 - np.max(bgrdash, axis=2)

# Calculate C
C = (1-bgrdash[...,2] - K)/(1-K)

# Calculate M
M = (1-bgrdash[...,1] - K)/(1-K)

# Calculate Y
Y = (1-bgrdash[...,0] - K)/(1-K)

# Combine 4 channels into single image and re-scale back up to uint8
CMYK = (np.dstack((C,M,Y,K)) * 255).astype(np.uint8)

print(type(Y))
print(Y.shape)
print(CMYK.shape)

cv2.imshow('CMYK Transformed',Y)
cv2.waitKey(0)
cv2.destroyAllWindows()
