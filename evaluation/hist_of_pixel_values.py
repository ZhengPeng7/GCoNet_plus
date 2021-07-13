import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


root_dir = 'gconet_24-ep82/CoCA/Accordion'
image_paths = [os.path.join(root_dir, p) for p in os.listdir(root_dir)]
pixel_values = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    pixel_value = image.flatten().squeeze().tolist()
    pixel_values += pixel_value

pixel_values = np.array(pixel_values)
non_zero_values = pixel_values[pixel_values > 0]
margin_values_percent = np.sum(non_zero_values > 230) / non_zero_values.shape[0] * 100
print('histing...')
plt.hist(x=non_zero_values)
plt.title('{:.1f} % are margin values'.format(margin_values_percent))
plt.savefig('hist.png')
