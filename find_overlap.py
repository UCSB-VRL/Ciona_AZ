from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import os
import ipdb


image_folder = 'Animal_1/cropped'
threshold = 0.9
similar_images = []
imnames = os.listdir(image_folder)

with open('storage/overlap.txt', 'w') as f:
    for i in range(len(imnames)):
        for j in range(i+1, len(imnames)):
            imname1 = os.path.join(image_folder, imnames[i])
            imname2 = os.path.join(image_folder, imnames[j])
            img1 = np.array(Image.open(imname1))
            img2 = np.array(Image.open(imname2))
            if len(img1.shape) > 2:
                img1 = img1[:,:,0]
            if len(img2.shape) > 2:
                img2 = img2[:,:,0]
            if img1.shape != img2.shape:
                ipdb.set_trace()
            s = ssim(img1, img2)
            #print(s)
            if s > threshold:
                print(imname1 + ' == ' + imname2)
                #ipdb.set_trace()
                similar_images.append((imname1, imname2))
                f.write(imname1)
                f.write('\n')
                f.write(imname2)
                f.write('\n')
