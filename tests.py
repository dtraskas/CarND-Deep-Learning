#
# Deep Learning Autonomous Car
# Small test library that needs more work
#
# Dimitrios Traskas
#
#
import os

# Loads all the image data from jpg files (used for testing)
def read_images(path):    
    sample_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])    
    image_array = np.ndarray((sample_count, HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)     
    cnt = 0
    for filename in os.listdir(path):
        fullFilename = os.path.join(path,filename)        
        image = read_image(fullFilename)
        if image is not None:
            image_array[cnt] = image
            cnt += 1    
    return image_array
