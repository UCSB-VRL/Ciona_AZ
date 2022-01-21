import torch
from PIL import Image
from skimage.transform import resize
import numpy as np
import ipdb
import os
import pickle


N = 500
subset = 'detection'

def preprocess(batch, mean, std):
	batch -= mean
	batch /= std
	return batch

image_folder = 'Animal_1/'
all_folders = ['cropped', 'non_synapse']


model = torch.load('../trials/resnext_50_pretrained.pt')


num_params = 0
for param in model.named_parameters():
	num_params += 1


print('num_params: ' + str(num_params))
#ipdb.set_trace()
model.fc = torch.nn.Linear(2048, len(all_folders), bias=False)
model.fc.requires_grad = True

reload_path = 'trials/2D_'+subset+'_trainvaltest_retrain=True_lr_0.0001_bs_2_epoch_200.pt'
model.load_state_dict(torch.load(reload_path))
dtype = torch.FloatTensor

if torch.cuda.is_available():
	dtype = torch.cuda.FloatTensor
	model.to('cuda')
	print('using cuda')
else:
	print('no cuda')
	dtype = torch.FloatTensor


#x = []
#y = []
index = 0

batch_size = 1



with open('storage/x_train_'+subset+'.pkl', 'rb') as f:
    x_train = pickle.load(f)
with open('storage/y_train_'+subset+'.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('storage/x_val_'+subset+'.pkl', 'rb') as f:
    x_val = pickle.load(f)
with open('storage/y_val_'+subset+'.pkl', 'rb') as f:
    y_val = pickle.load(f)
with open('storage/x_test_'+subset+'.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('storage/y_test_'+subset+'.pkl', 'rb') as f:
    y_test = pickle.load(f)
total_mean = np.load('storage/mean_'+subset+'.npy')
total_std = np.load('storage/std_'+subset+'.npy')




print('x_test len: ' + str(len(x_test)))
print('y_test len: ' + str(len(y_test)))

print('x_val len: ' + str(len(x_val)))
print('y_val len: ' + str(len(y_val)))

x = x_train
y = y_train

print('x_train len: ' + str(len(x)))
print('y_train len: ' + str(len(y)))

#ipdb.set_trace()

def ciona_data_gen():
	N_train = len(x)
	X = np.empty([batch_size,3,N,N])
	Y = np.empty([batch_size, len(all_folders)])
	while True:
		#print(N_train)
		inds = np.random.permutation(N_train)
		i=0
		while i <= (N_train - batch_size):
			for j in range(batch_size):
				im_name = x[inds[i+j]]
				#print(im_name)
				img = np.array(Image.open(im_name))
				if len(img.shape)>2:
					img = img[:,:,0]
				X[j] = img
				Y[j] = y[inds[i+j]]
			X = preprocess(X, total_mean, total_std)
			i+=batch_size
			yield im_name, torch.from_numpy(X).type(dtype), torch.from_numpy(Y).type(dtype)


def ciona_data_gen_val():
	N_train = len(x_val)
	X = np.empty([batch_size,3,N,N])
	Y = np.empty([batch_size, len(all_folders)])
	while True:
		#print(N_train)
		inds = np.random.permutation(N_train)
		i=0
		while i <= (N_train - batch_size):
			for j in range(batch_size):
				im_name = x_val[inds[i+j]]
				#print(im_name)
				img = np.array(Image.open(im_name))
				if len(img.shape)>2:
					img = img[:,:,0]
				X[j] = img
				Y[j] = y_val[inds[i+j]]
			X = preprocess(X, total_mean, total_std)
			i+=batch_size
			yield im_name, torch.from_numpy(X).type(dtype), torch.from_numpy(Y).type(dtype)


def ciona_data_gen_test():
	N_train = len(x_test)
	X = np.empty([batch_size,3,N,N])
	Y = np.empty([batch_size, len(all_folders)])
	while True:
		#print(N_train)
		inds = np.random.permutation(N_train)
		i=0
		while i <= (N_train - batch_size):
			for j in range(batch_size):
				im_name = x_test[inds[i+j]]
				#print(im_name)
				img = np.array(Image.open(im_name))
				if len(img.shape)>2:
					img = img[:,:,0]
				X[j] = img
				Y[j] = y_test[inds[i+j]]
			X = preprocess(X, total_mean, total_std)
			i+=batch_size
			yield im_name, torch.from_numpy(X).type(dtype), torch.from_numpy(Y).type(dtype)



data_gen = ciona_data_gen()
data_gen_val = ciona_data_gen_val()
data_gen_test = ciona_data_gen_test()

with open('failed_cases.txt', 'w') as f:

    model.eval()
    with torch.no_grad():
        running_accuracy = 0
        for m in range(int(len(x_train))):
            name, inputs, labels = next(data_gen)
            outputs = model(inputs)
            labels_np = labels.cpu().numpy()
            outputs_np = outputs.cpu().detach().numpy()
            accuracy = np.mean(np.argmax(outputs_np,1) == np.argmax(labels_np,1))
            if accuracy == 0:
                f.write(name)
                f.write('\n')
            running_accuracy += accuracy
        avg_accuracy = running_accuracy / int(len(x_train)/batch_size)
        print('avg train accuracy: ' + str(avg_accuracy))

        running_accuracy = 0
        for m in range(int(len(x_val))):
            name, inputs, labels = next(data_gen_val)
            outputs = model(inputs)
            labels_np = labels.cpu().numpy()
            outputs_np = outputs.cpu().detach().numpy()
            accuracy = np.mean(np.argmax(outputs_np,1) == np.argmax(labels_np,1))
            if accuracy == 0:
                f.write(name)
                f.write('\n')
            running_accuracy += accuracy
        avg_accuracy = running_accuracy / int(len(x_val)/batch_size)
        print('avg validation accuracy: ' + str(avg_accuracy))

        running_accuracy = 0
        for m in range(int(len(x_test))):
            name, inputs, labels = next(data_gen_test)
            outputs = model(inputs)
            labels_np = labels.cpu().numpy()
            outputs_np = outputs.cpu().detach().numpy()
            accuracy = np.mean(np.argmax(outputs_np,1) == np.argmax(labels_np,1))
            if accuracy == 0:
                f.write(name)
                f.write('\n')
            running_accuracy += accuracy
        avg_accuracy = running_accuracy / int(len(x_test)/batch_size)
        print('avg test accuracy: ' + str(avg_accuracy))

