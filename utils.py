import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def pre_process1(image, top_prop=0.35, bottom_prop=0.1):
    """
        - Crop the top `top_prop` and the bottom `bottom_prop` of the image
        - Resize the image to half of it's original size
    """
    rows_to_crop_top = int(image.shape[0] * 0.4)
    rows_to_crop_bottom = int(image.shape[0] * 0.1)
    image = image[rows_to_crop_top:image.shape[0] - rows_to_crop_bottom, :]
    print("images shape inutlity",image.shape)

    reduceImage=cv2.resize(image, (0,0), fx=0.5, fy=0.5)
    print("images reduce shape inutlity", reduceImage.shape)
    plt.figure()
    plt.imshow(reduceImage)
    return reduceImage
	
def pre_process(image, top_prop=0.35, bottom_prop=0.1):
	rows_to_crop_top = int(image.shape[0] * 0.4)
	rows_to_crop_bottom = int(image.shape[0] * 0.1)
	image = image[rows_to_crop_top:image.shape[0] - rows_to_crop_bottom, :]

	reduceImage = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
	return reduceImage
