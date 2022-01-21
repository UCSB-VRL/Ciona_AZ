import torch
from PIL import Image
from skimage.transform import resize
from collections import Counter
import numpy as np
import ipdb
import os
import pickle


N = 500
do_postsynaptic = True
total_batch_size=16

def preprocess(batch, mean, std):
    batch -= mean
    batch /= std
    return batch

image_folder = 'Animal_1/cropped'
all_folders = [image_folder]
num_classes = 2
#classes = ['inhibitory', 'excitatory']
model = torch.load('../trials/resnext_pretrained.pt')
#subset = 'inhibitory_excitatory'
id_to_im = pickle.load(open('storage/id_to_im.pkl', 'rb'))
pre_to_im = pickle.load(open('storage/pre_to_im.pkl','rb'))
reload_stats = True
use_dict = False
redo = False
ID_pred_path = 'storage/ID_to_pred.pkl'
pre_pred_path = 'storage/pre_to_pred.pkl'


Gly_folder = 'gly'
ACh_folder = 'ach'
GABA_folder = 'gaba'
Glut_folder = 'glut'
ddn_folder = 'ddn'
mgin_folder = 'mgin'
antrn_folder = 'antrn'
prrn_folder = 'pr-amgrn'
#all_folders = ['ach', 'gaba', 'glut', 'gly']

#subset = 'ach_gaba'
subset = 'excitatory_inhibitory'


if os.path.exists(ID_pred_path) and not redo:
    print('preloading dict')
    ID_to_pred = pickle.load(open(ID_pred_path, 'rb'))
else:
    ID_to_pred = {}

if use_dict:
    with open('storage/synapse_trainval_'+subset+'.pkl', 'rb') as f:
        syn_dict = pickle.load(f)
    train_val = {'train':[], 'val':[]}
    for key in syn_dict.keys():
        train_val[syn_dict[key]].append(key)


for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(2048, num_classes, bias=False)

#reload_path = 'trials/2D_inhibitory_excitatory_retrain=True_lr_0.0001_bs_4_epoch_150.pt'
#reload_path = 'trials/2D_inhibitory_excitatory_trainvaltest_retrain=True_lr_0.0001_bs_4_epoch_200.pt'
reload_path = 'trials/2D_inhibitory_excitatory_trainvaltest_retrain=True_lr_0.0001_bs_2_epoch_200.pt'

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


for folder in all_folders:
    ind = 0
    for imname in os.listdir(folder):
        im_name = os.path.join(folder, imname)
        if os.path.isdir(im_name) or not im_name.endswith('.tif'):
            continue
        x.append(im_name)
        ind += 1
    index += 1

total_batch_size = 50
x_new = x.copy()

if reload_stats:
    total_mean = np.load('storage/mean.npy')
    total_std = np.load('storage/std.npy')
else:
    stats_stack = [0,0]
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
        stats_stack[0] += np.mean(img)
        stats_stack[1] += np.std(img)

    del x_new

    total_mean = stats_stack[0]/len(x)
    total_std = stats_stack[1]/len(x)
    print('mean: ' + str(total_mean))
    print('std: ' + str(total_std))
    np.save('storage/mean.npy', total_mean)
    np.save('storage/std.npy', total_std)

id_keys = list(id_to_im.keys())

N_train = len(x)
del x

done_images = set()

def ciona_data_gen():
    curr_key = 0
    curr_im = 0
    while curr_im < N_train:
        curr_ID = id_keys[curr_key]
        curr_ims = id_to_im[curr_ID]
        curr_ims = list(curr_ims)
        batch_size = len(curr_ims)
        X = np.empty([batch_size,3,N,N])
        im_names = []
        for j in range(batch_size):
            im_name = curr_ims[j]
            im_names.append(im_name)
            #print(im_name)
            im_name = os.path.join(image_folder, im_name)
            img = np.array(Image.open(im_name))
            #img = resize(img, output_shape=[N,N],preserve_range=True,mode='constant',order=1)
            if len(img.shape) > 2:
                img = img[:,:,0]
            img = np.stack([img,img,img],axis=0)
            X[j] = img
        X = preprocess(X, total_mean, total_std)
        curr_key += 1
        curr_im += batch_size
        yield curr_ID, im_names, X

        
def ciona_data_gen_pre():
    curr_key = 0
    while True:
        pre_name = pre_keys[curr_key]
        pre_name_syn = pre_to_im[pre_name]
        curr_ims = []
        for synapse_name in pre_name_syn.keys():
            images = pre_name_syn[synapse_name]
            images = list(images)
            curr_ims += images
        batch_size = len(curr_ims)
        X = np.empty([batch_size,3,N,N])
        im_names = []
        for j in range(batch_size):
            im_name = curr_ims[j]
            im_names.append(im_name)
            #print(im_name)
            im_name = os.path.join(image_folder, im_name)
            img = np.array(Image.open(im_name))
            #img = resize(img, output_shape=[N,N],preserve_range=True,mode='constant',order=1)
            if len(img.shape) > 2:
                img = img[:,:,0]
            img = np.stack([img,img,img],axis=0)
            X[j] = img
        X = preprocess(X, total_mean, total_std)
        curr_key += 1
        yield pre_name, len(pre_name_syn.keys()), im_names, X   

        
data_gen = ciona_data_gen()

with torch.no_grad():
    model.eval()
    print('# preloaded IDs: ' + str(len(ID_to_pred.keys())))
    #ipdb.set_trace()
    ''' Prediction '''
    if do_postsynaptic:
        for k in range(len(id_keys)):
            ID, names, inputs = next(data_gen)
            #ipdb.set_trace()
            if ID in ID_to_pred.keys():
                print('skipped')
                del inputs
                del names
                del ID
                continue
            total_outputs = []
            #ipdb.set_trace()
            while len(total_outputs) < len(names)/total_batch_size:
                start_point = len(total_outputs)*total_batch_size
                curr_inputs = inputs[start_point:start_point+total_batch_size]
                curr_inputs = torch.from_numpy(curr_inputs).type(dtype)
                curr_outputs = model(curr_inputs)
                total_outputs.append(curr_outputs.cpu().detach().numpy())
                del curr_outputs
                del curr_inputs
            outputs_np = np.concatenate(total_outputs)
            print(ID)
            predictions = np.argmax(outputs_np, 1)
            occurence_count = Counter(predictions)
            estimate=occurence_count.most_common(1)[0][0]
            #estimate = predictions.sum()/predictions.size
            certainty = np.mean(np.abs(outputs_np)[:,0])
            ID_to_pred[ID] = (estimate, certainty, len(names))
            del inputs
            del names
            del ID
            #ipdb.set_trace()

        pickle.dump(ID_to_pred, open(ID_pred_path, 'wb'))
        #ipdb.set_trace()
        print('done with ID pred')
    else:
        pre_keys = list(pre_to_im.keys())
        data_gen_pre = ciona_data_gen_pre()
        pre_to_pred = {}

        for k in range(len(pre_keys)):
            prename, num_synapses, names, inputs = next(data_gen_pre)
            total_outputs = []
            #ipdb.set_trace()
            while len(total_outputs) < len(names)/total_batch_size:
                start_point = len(total_outputs)*total_batch_size
                curr_inputs = inputs[start_point:start_point+total_batch_size]
                curr_inputs = torch.from_numpy(curr_inputs).type(dtype)
                curr_outputs = model(curr_inputs)
                total_outputs.append(curr_outputs.cpu().detach().numpy())
                del curr_outputs
                del curr_inputs
            outputs_np = np.concatenate(total_outputs)
            print(prename)
            predictions = np.argmax(outputs_np, 1)
            #ipdb.set_trace()
            occurence_count = Counter(predictions)
            estimate=occurence_count.most_common(1)[0][0]
            #estimate = predictions.sum()/predictions.size
            certainty = np.mean(np.abs(outputs_np)[:,0])
            if prename in pre_to_pred.keys():
                pre_to_pred[prename].append((estimate, certainty, num_synapses, len(names)))
            else:
                pre_to_pred[prename] = [(estimate, certainty, num_synapses, len(names))]

        pickle.dump(pre_to_pred, open(pre_pred_path, 'wb'))
        print('done with pred')