import torch
from PIL import Image
from skimage.transform import resize
import numpy as np
#import ipdb
import os
import pickle


N = 500
do_postsynaptic = True

def preprocess(batch, mean, std):
	batch -= mean
	batch /= std
	return batch


image_folder = 'Animal_1/pr_sorted/group2'
#image_folder = 'Animal_1/rn_sorted/antrn'
#image_folder = 'Animal_1/pr_sorted/glut'
#image_folder = 'Animal_1/cropped'
all_folders = [image_folder]
num_classes = 2
classes = ['inhibitory', 'excitatory']
model = torch.load('../trials/resnext_pretrained.pt')
#subset = 'inhibitory_excitatory'
subset = 'pr'
#use_dict = True
use_dict = False


if use_dict:
	with open('synapse_trainval_'+subset+'.pkl', 'rb') as f:
		syn_dict = pickle.load(f)
	train_val = {'train':[], 'val':[]}
	for key in syn_dict.keys():
		train_val[syn_dict[key]].append(key)


for param in model.parameters():
	param.requires_grad = False

model.fc = torch.nn.Linear(2048, num_classes, bias=False)

#reload_path = 'trials/2D_inhibitory_excitatory_retrain=True_lr_0.0001_bs_4_epoch_200.pt'
reload_path = 'trials/2D_pr_retrain=True_lr_0.0001_bs_4_epoch_100.pt'
model.load_state_dict(torch.load(reload_path))

if torch.cuda.is_available():
	dtype = torch.cuda.FloatTensor
	model.to('cuda')
	print('using cuda')
else:
	print('no cuda')
	dtype = torch.FloatTensor


x = []
index = 0


f_lens = []
for folder in all_folders:
	ind = 0
	unique_names = set([f[:-6] for f in os.listdir(folder) if not os.path.isdir(os.path.join(folder,f))])
	f_len = len(unique_names)
	f_lens.append(f_len)

f_len = min(f_lens)

for folder in all_folders:
	ind = 0
	for imname in os.listdir(folder):
		im_name = os.path.join(folder, imname)
		if os.path.isdir(im_name) or not im_name.endswith('.tif'):
			continue
		x.append(im_name)
		ind += 1
	index += 1

batch_size = 1
x_new = x.copy()

stats_stack = []
for i in range(len(x)):
	im_name = x[i]
	img = np.array(Image.open(im_name))
	if len(img.shape) < 2:
		print('image name: ' + im_name)
		print('image shape: ' + str(img.shape))
		x.remove(im_name)
		continue
	elif not (img.shape[0] == 500 and img.shape[1] == 500):
		print('image name: ' + im_name)
		print('image shape: ' + str(img.shape))
		x_new.remove(im_name)
		continue
	elif len(img.shape) > 2:
		img = img[:,:,0]
	#img = resize(img, output_shape = [N,N], preserve_range=True, mode = 'constant', order = 1)
	stats_stack.append(img)

x = x_new.copy()
del x_new

with open('x_pred.pkl', 'wb') as f:
	pickle.dump(x, f)

total_mean = np.mean(stats_stack)
total_std = np.std(stats_stack)
print('mean: ' + str(total_mean))
print('std: ' + str(total_std))
np.save('mean.npy', total_mean)
np.save('std.npy', total_std)


def ciona_data_gen():
	N_train = len(x)
	X = np.empty([batch_size,3,N,N])
	while True:
		#print(N_train)
		inds = np.random.permutation(N_train)
		inds = np.sort(inds)
		#print(inds)
		i=0
		while i <= (N_train - batch_size):
			for j in range(batch_size):
				index = inds[i+j]
				im_name = x[index]
				#print(im_name)
				img = np.array(Image.open(im_name))
				#img = resize(img, output_shape=[N,N],preserve_range=True,mode='constant',order=1)
				if len(img.shape) > 2:
					img = img[:,:,0]
				img = np.stack([img,img,img],axis=0)
				X[j] = img
			X = preprocess(X, total_mean, total_std)
			i+=batch_size
			yield im_name, torch.from_numpy(X).type(dtype)


data_gen = ciona_data_gen()
model.eval()
predictions = []
confidence = []
names = []
pred_dict = {}
post_dict = {}
pre_post_dict = {}
if use_dict:
	train_val_dict = {'train': [], 'val': []}

''' Prediction '''
for k in range(int(len(x)/batch_size)):
	name, inputs = next(data_gen)
	outputs = model(inputs)
	outputs_np = outputs.cpu().detach().numpy()
	#print(outputs_np)
	prediction = np.argmax(outputs_np, 1)[0]
	name = os.path.split(name)[-1]
	if use_dict:
		synapse_id = '_'.join(name.split('_')[1:])
		if synapse_id in train_val['train']:
			train_val_dict['train'].append(prediction)
		else:
			train_val_dict['val'].append(prediction)
	predictions.append(prediction)
	confidence.append(outputs_np)
	names.append(name)
	#ipdb.set_trace()

	if do_postsynaptic:
		f = name.replace('}', '-')
		f = f.replace('_', '-')
		namelist = f.split('-')
		while ' ' in namelist:
			namelist.remove(' ')
		while '' in namelist:
			namelist.remove('')
		#ipdb.set_trace()
		if len(namelist) < 5:
			continue
		if 'pr' in name.split('-')[0]:
			postsyn = namelist[3]
			if postsyn == 'pr':
				postsyn = postsyn + namelist[4]
		else:
			postsyn = namelist[2]
			if postsyn == 'pr':
				postsyn = postsyn + namelist[4]

	#ipdb.set_trace()
	if 'pr' in name.split('-')[0]:
		name = name[name.find('pr'):name.find('pr')+4]
		name = ''.join(ch for ch in name if ch.isalnum())
		new_name = name
		for c in range(len(name)):
			char = name[c]
			if name[2].isnumeric() and c > 2:
				if not char.isnumeric():
					del new_name[c]
	else:
		new_name = name[name.find('00syn'):name.find('-')][5:]
	name = new_name
	if name in pred_dict:
		pred_dict[name].append(prediction)
	else:
		pred_dict[name] = [prediction]
	if do_postsynaptic:
		#print(postsyn)
		if postsyn in post_dict:
			post_dict[postsyn].append(prediction)
		else:
			post_dict[postsyn] = [prediction]
		pre_post = name+'_'+postsyn
		if pre_post in pre_post_dict:
			pre_post_dict[pre_post].append(prediction)
		else:
			pre_post_dict[pre_post] = [prediction]

if use_dict:
	val_acc = sum(train_val_dict['val']) / len(train_val_dict['val'])
	train_acc = sum(train_val_dict['train']) / len(train_val_dict['train'])
	print('val acc: ' + str(val_acc))
	print('train acc: ' + str(train_acc))

with open('predictions.pkl', 'wb') as f:
	pickle.dump(predictions, f)

with open('names.pkl', 'wb') as f:
	pickle.dump(names, f)

with open('pred_dict.pkl', 'wb') as f:
	pickle.dump(pred_dict, f)

if do_postsynaptic:
	with open('post_dict.pkl', 'wb') as f:
		pickle.dump(post_dict, f)

#ipdb.set_trace()



