import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(path):
   
   
    img = Image.open(path)
   
    return np.array(img)

def edge_detection(image):
    
    
    if len(image.shape) == 3:
        # Axis 2 is the color channel axis. Averaging collapses it to 2D.
        gray_image = np.mean(image, axis=2)
    else:
        # If already grayscale, just use it as is
        gray_image = image

    # Step 2: Define the filters (kernels) from the instructions
    # KernelY: Detects changes in the vertical direction
    kernelY = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    # KernelX: Detects changes in the horizontal direction
    kernelX = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

    # Step 3: Apply convolution
    # mode='same' ensures the output size matches the input size
    # boundary='fill', fillvalue=0 ensures zero padding is used
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill', fillvalue=0)

    # Step 4: Compute the magnitude
    # Formula: sqrt(edgeX^2 + edgeY^2)
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG
