import numpy as np
from PIL import Image
from skimage.transform import resize
from tifffile import imread, imsave, imshow
from matplotlib import pyplot as plt
import os
import pickle
import ipdb

folder = 'Animal_1/Synapse_data'
dictpath = 'storage/resolution_dict.pkl'
newpath = 'Animal_1/cropped'
resdict = pickle.load(open(dictpath, 'rb'))
resolutions = {'11500':564/2, '16500':267.0, '9900':319/2, '8200':492/2, '6000':354/2, '26500':375/0.5, '20500':290/0.5}

#ipdb.set_trace()

for imname in os.listdir(folder):
	dictname = '_'.join(imname.split('_')[1:])
	dictname = dictname[:dictname.find('.tif')]
	if dictname not in resdict.keys():
		continue
	impath = os.path.join(folder, imname)
	if not os.path.exists(impath):
		continue
	try:
		img = Image.open(impath)
		resolution = resdict[dictname]
		ratio = resolution / 159.5
		orig_size = img.size[0]
		N = int(orig_size * ratio)
		img = img.resize((N, N), Image.LANCZOS)
		new_size = img.size[0]
		print('new size: ' + str(new_size))
		cut = int((new_size - orig_size)/2)
		img = img.crop((cut, cut, new_size-cut, new_size-cut))
		img = img.crop((0, 0, orig_size, orig_size))
		#img.show()
		newname = os.path.join(newpath, imname)
		img.save(newname)
		#ipdb.set_trace()
	except:
		continue
