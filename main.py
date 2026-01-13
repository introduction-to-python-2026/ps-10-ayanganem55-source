import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball

# Import the functions we created in the other file
from image_utils import load_image, edge_detection

def main():
    # 1. Load the image (REPLACE 'my_image.jpg' WITH YOUR IMAGE FILE NAME)
    image_path = 'my_image.jpg' 
    try:
        original_image = load_image(image_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{image_path}'. Please add an image file.")
        return

    print("Image loaded successfully.")

    # 2. Suppress noise using a median filter
    # ball(3) creates a 3D footprint radius 3. 
    # This cleans 'salt and pepper' noise before edge detection.
    clean_image = median(original_image, ball(3))
    print("Noise suppression complete.")

    # 3. Detect edges using our custom function
    edge_mag = edge_detection(clean_image)
    print("Edge detection complete.")

    # 4. Thresholding to create a binary image
    # You can view the histogram to pick a better number, but 50-100 is usually a good start.
    # Values > threshold become 1 (White edge), others become 0 (Black background).
    threshold = 80
    binary_edges = edge_mag > threshold

    # 5. Save the result
    # Convert boolean (True/False) to uint8 (0-255) for saving as an image
    final_image_array = (binary_edges * 255).astype(np.uint8)
    edge_image = Image.fromarray(final_image_array)
    
    output_filename = 'my_edges.png'
    edge_image.save(output_filename)
    print(f"Edge image saved as '{output_filename}'.")

    # Optional: Display images to verify
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(final_image_array, cmap='gray')
    plt.title("Edge Detection")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
