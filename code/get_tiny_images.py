from PIL import Image

import pdb
import numpy as np

def get_tiny_images(image_paths):

    '''
    Input : 
        image_paths: a list(N) of string where where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    
    N = len(image_paths)
    size = 16
    
    tiny_images = []
    
    for each in image_paths:
        image = Image.open(each)
        image = image.resize((size, size))
        image = (image - np.mean(image))/np.std(image)
        image = image.flatten()
        tiny_images.append(image)
        
    tiny_images = np.asarray(tiny_images)
    #print(tiny_images.shape)
    
    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images