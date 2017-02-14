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

#%%
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv( "track_one_data/driving_log.csv", header=None, names=['center', 'left', 'right', 'angle', 'throttle', 'break', 'speed'])
s1 = data['angle']
plt.figure()
s1.plot.hist(alpha=0.5)
plt.savefig('my_hist.jpg')

#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.misc import imread
from skimage import transform
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid

def add_image(image, ax, title):    
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title)    

def load_image(pd, pos):       
    filename = pd[pos].values[0]
    image = transform.resize(imread("track_one_data/IMG/" + filename.rsplit("/")[-1]), (80,160))
    return image

def read_image(filename):
    return transform.resize(imread("track_one_data/IMG/" + filename.rsplit("/")[-1]), (80,160))

data = pd.read_csv( "track_one_data/driving_log.csv", header=None, names=['center', 'left', 'right', 'angle', 'throttle', 'break', 'speed'])

fig = plt.figure(1, (10, 10))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, 3),  # creates 2x2 grid of axes                
                 axes_pad=0.3  # pad between axes in inch.
                 )
d1 = data.sample(n=1)    


img1 = read_image(' /Users/talos/Projects/Udacity/DL-Autonomous-Car/track_one_data/IMG/left_2017_02_12_22_15_38_042.jpg')

img2 = read_image('/Users/talos/Projects/Udacity/DL-Autonomous-Car/track_one_data/IMG/center_2017_02_12_22_12_37_042.jpg')

img3 = read_image(' /Users/talos/Projects/Udacity/DL-Autonomous-Car/track_one_data/IMG/right_2017_02_12_22_20_47_042.jpg')

add_image(img1, grid[0], 'Left camera (steering = ' + str(round(-0.08012585, 3)) + ')')
add_image(img2, grid[1], 'Center camera (steering = ' + str(round(-0.2931493, 3))  + ')')
add_image(img3, grid[2], 'Right camera (steering = ' + str(round(0, 3))  + ')')
plt.savefig('images_two.jpg')

'''
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 3),  # creates 2x2 grid of axes                
                 axes_pad=0.3  # pad between axes in inch.
                 )

for i in range(0, 3):
    d1 = data.sample(n=1)    
    add_image(load_image(d1,'center'), grid[i * 3], d1['angle'].values)
    d1 = data.sample(n=1)    
    add_image(load_image(d1,'left'), grid[i * 3 + 1], d1['angle'].values)
    d1 = data.sample(n=1)    
    add_image(load_image(d1,'right'), grid[i * 3 + 2], d1['angle'].values)
'''

#plt.savefig('images.jpg')
plt.show()
