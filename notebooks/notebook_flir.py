
#%%
# # dead_simple.py
from PIL import Image
import os
from simple_pyspin import Camera

with Camera() as cam: # Acquire and initialize Camera
    cam.start() # Start recording
    imgs = [cam.get_array() for n in range(10)] # Get 10 frames
    cam.stop() # Stop recording

print(imgs[0].shape, imgs[0].dtype) # Each image is a numpy array!


# Make a directory to save some images
output_dir = 'test_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving images to: %s" % output_dir)

# Save them
# NOTE: images may be very dark or bright, depending on the camera lens and
#   room conditions!
for n, img in enumerate(imgs):
    Image.fromarray(img).save(os.path.join(output_dir, '%08d.png' % n))