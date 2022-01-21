import torch
from PIL import Image
from skimage.transform import resize
import numpy as np
import ipdb
import os
import pickle


N = 500
retrain = True
do_reload = False
epoch = 60
subset = 'ach_gaba'
#subset = 'pr'
use_subset = False

overlap_F = open('storage/overlap.txt', 'r')
overlap_list = overlap_F.read().splitlines()
overlap_F.close()
im_to_pre = pickle.load(open('storage/im_to_pre.pkl', 'rb'))
train_split = 0.8
test_split = 0.1
val_split = 0.1

def preprocess(batch, mean, std):
    batch -= mean
    batch /= std
    return batch


if use_subset:
    image_folder = 'Animal_1/'+subset+'_sorted'
else:
    image_folder = 'Animal_1/'
Gly_folder = 'gly'
ACh_folder = 'ach'
GABA_folder = 'gaba'
Glut_folder = 'glut'
ddn_folder = 'ddn'
mgin_folder = 'mgin'
antrn_folder = 'antrn'
prrn_folder = 'pr-amgrn'
#inhibitory_folder = 'GABA_and_Gly'
#all_folders = [Gly_folder, ACh_folder, GABA_folder, Glut_folder]
#all_folders = [GABA_folder, Glut_folder]
all_folders = ['ach', 'gaba']
if use_subset:
    if subset == 'mg':
        all_folders = [ddn_folder, mgin_folder]
    elif subset == 'rn':
        all_folders = [antrn_folder, prrn_folder]

model = torch.load('../trials/resnext_50_pretrained.pt')

#conv1_weights = model.conv1.weight.data
#model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
#model.conv1.weight.data = conv1_weights[:,0:1,:,:]


num_params = 0
for param in model.named_parameters():
    if retrain:
        #print(param[0])
        param[1].requires_grad = True
    else:
        param[1].requires_grad = False
    num_params += 1


print('num_params: ' + str(num_params))
#ipdb.set_trace()
model.fc = torch.nn.Linear(2048, len(all_folders), bias=False)
model.fc.requires_grad = True

if retrain:
    #reload_path = 'trials/2D_pr_retrain=True_lr_0.0001_bs_4_epoch_100.pt'
    reload_path = 'trials/2D_'+subset+'_trainvaltest_retrain=False_lr_0.0001_bs_4_epoch_190.pt'
    #reload_path = 'trials/2D_inhibitory_excitatory_retrain=True_lr_0.0001_bs_4_epoch_150.pt'
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

batch_size = 4
#batch_size = 1
if retrain:
    num_epochs = 300
else:
    num_epochs = 200


if retrain:
    print('retrain')
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
else:
    im_to_synapse = {}
    im_to_cell = {}
    type_to_cell = {key:set() for key in all_folders}
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    f_lens = []
    for folder in all_folders:
        folder = os.path.join(image_folder, folder)
        ind = 0
        unique_names = set([f[:-6] for f in os.listdir(folder) if not os.path.isdir(os.path.join(folder,f))])
        f_len = len(unique_names)
        f_lens.append(f_len)

    #ipdb.set_trace()
    #f_len = min(f_lens)

    for f in all_folders:
        folder = os.path.join(image_folder, f)
        ind = 0
        f_len = f_lens[index]
        for imname in os.listdir(folder):
            if imname in overlap_list:
                ind += 1
                #print('skipped due to overlap')
                continue
            synapse_id = '_'.join(imname.split('_')[1:]).lower()
            im_name = os.path.join(folder, imname)
            if os.path.isdir(im_name) or ind > f_len:
                continue
            if imname not in im_to_pre.keys():
                print('not found in dict: ' + imname)
                continue
            cellID = im_to_pre[imname]
            if cellID in im_to_synapse.keys():
                if synapse_id in im_to_synapse[cellID].keys():
                    im_to_synapse[cellID][synapse_id].append(im_name)
                else:
                    im_to_synapse[cellID][synapse_id] = [im_name]
            else:
                im_to_synapse[cellID] = {synapse_id: [im_name]}
            im_to_cell[imname] = cellID
            type_to_cell[f].add(cellID)
            ind += 1
        index += 1
    
    pickle.dump(im_to_synapse, open('storage/im_to_synapse.pkl', 'wb'))
    pickle.dump(im_to_cell, open('storage/im_to_cell.pkl', 'wb'))
    
    index = 0
    done_cells = set()
    test_cells = set()
    train_cells = set()
    for t in type_to_cell.keys():
        cells = list(type_to_cell[t])
        cell_inds = np.random.permutation(len(cells))
        ind_1 = int(train_split*len(cells))
        ind_2 = ind_1 + int(test_split*len(cells))
        ind_3 = min(len(cells), ind_2 + int(val_split*len(cells)))
        train_inds = cell_inds[0:ind_1]
        test_inds = cell_inds[ind_1:ind_2]
        val_inds = cell_inds[ind_2:ind_3]
        for train_ind in train_inds:
            cell = cells[train_ind]
            if cell in done_cells:
                print('done cell')
                ipdb.set_trace()
            done_cells.add(cell)
            train_cells.add(cell)
            synapses = im_to_synapse[cell]
            for synapse in synapses.keys():
                x_train += synapses[synapse]
                for s in synapses[synapse]:
                    y_temp = [0]*len(all_folders)
                    y_temp[index] = 1
                    y_train.append(y_temp)
                
        for test_ind in test_inds:
            cell = cells[test_ind]
            if cell in done_cells:
                print('done cell')
                ipdb.set_trace()
            done_cells.add(cell)
            test_cells.add(cell)
            synapses = im_to_synapse[cell]
            for synapse in synapses.keys():
                x_test += synapses[synapse]
                for s in synapses[synapse]:
                    y_temp = [0]*len(all_folders)
                    y_temp[index] = 1
                    y_test.append(y_temp)
                
        for val_ind in val_inds:
            cell = cells[val_ind]
            if cell in done_cells:
                print('done cell')
                ipdb.set_trace()
            done_cells.add(cell)
            test_cells.add(cell)
            synapses = im_to_synapse[cell]
            for synapse in synapses.keys():
                x_val += synapses[synapse]
                for s in synapses[synapse]:
                    y_temp = [0]*len(all_folders)
                    y_temp[index] = 1
                    y_val.append(y_temp)
        index += 1
    pickle.dump(test_cells, open('storage/test_cells.pkl', 'wb'))
    pickle.dump(train_cells, open('storage/train_cells.pkl', 'wb'))
    
    stats_stack = []
    to_remove = []
    for i in range(len(x_train)):
        im_name = x_train[i]
        img = np.array(Image.open(im_name))
        #print(img.shape)
        if len(img.shape) < 2:
            print('image name: ' + im_name)
            print('image shape: ' + str(img.shape))
            to_remove.append(i)
        elif len(img.shape) > 2:
            img = img[:,:,0]
        if not (img.shape[0] == 500 and img.shape[1] == 500):
            print('image name: ' + im_name)
            print('image shape: ' + str(img.shape))
            to_remove.append(i)
        #img = resize(img, output_shape = [N,N], preserve_range=True, mode = 'constant', order = 1)
        else:   
            stats_stack.append(img)


    for i in range(len(to_remove)):
        count = to_remove[i]
        del x_train[count]
        del y_train[count]
        for j in range(i, len(to_remove)):
            to_remove[j] -= 1

    to_remove = []
    for i in range(len(x_val)):
        im_name = x_val[i]
        img = np.array(Image.open(im_name))
        if len(img.shape) < 2:
            print('image name: ' + im_name)
            print('image shape: ' + str(img.shape))
            to_remove.append(i)
        elif len(img.shape) > 2:
            img = img[:,:,0]
        if not (img.shape[0] == 500 and img.shape[1] == 500):
            print('image name: ' + im_name)
            print('image shape: ' + str(img.shape))
            to_remove.append(i)

    for i in range(len(to_remove)):
        count = to_remove[i]
        del x_val[count]
        del y_val[count]
        for j in range(i, len(to_remove)):
            to_remove[j] -= 1

    to_remove = []
    for i in range(len(x_test)):
        im_name = x_test[i]
        img = np.array(Image.open(im_name))
        if len(img.shape) < 2:
            print('image name: ' + im_name)
            print('image shape: ' + str(img.shape))
            to_remove.append(i)
        elif len(img.shape) > 2:
            img = img[:,:,0]
        if not (img.shape[0] == 500 and img.shape[1] == 500):
            print('image name: ' + im_name)
            print('image shape: ' + str(img.shape))
            to_remove.append(i)

    for i in range(len(to_remove)):
        count = to_remove[i]
        del x_test[count]
        del y_test[count]
        for j in range(i, len(to_remove)):
            to_remove[j] -= 1


    with open('storage/x_train_'+subset+'.pkl', 'wb') as f:
        pickle.dump(x_train, f)
    with open('storage/y_train_'+subset+'.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('storage/x_val_'+subset+'.pkl', 'wb') as f:
        pickle.dump(x_val, f)
    with open('storage/y_val_'+subset+'.pkl', 'wb') as f:
        pickle.dump(y_val, f)
    with open('storage/x_test_'+subset+'.pkl', 'wb') as f:
        pickle.dump(x_test, f)
    with open('storage/y_test_'+subset+'.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    with open('storage/synapse_trainvaltest_'+subset+'.pkl', 'wb') as f:
        pickle.dump(im_to_synapse, f)

    
    total_mean = np.mean(stats_stack)
    total_std = np.std(stats_stack)
    print('mean: ' + str(total_mean))
    print('std: ' + str(total_std))
    np.save('storage/mean_'+subset+'.npy', total_mean)
    np.save('storage/std_'+subset+'.npy', total_std)


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
            yield torch.from_numpy(X).type(dtype), torch.from_numpy(Y).type(dtype)


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
            yield torch.from_numpy(X).type(dtype), torch.from_numpy(Y).type(dtype)


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
            yield torch.from_numpy(X).type(dtype), torch.from_numpy(Y).type(dtype)



data_gen = ciona_data_gen()
data_gen_val = ciona_data_gen_val()
data_gen_test = ciona_data_gen_test()

''' Training '''
criterion = torch.nn.BCEWithLogitsLoss()
lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=0.0001)
r = range(num_epochs)
print_every = 5
if do_reload:
    reload_path = 'trials/2D_'+subset+'_trainvaltest_retrain=False_lr_0.0001_bs_4_epoch_' + str(epoch) + '.pt'
    #reload_path = 'trials/2D_inhibitory_excitatory_retrain=True_lr_0.0001_bs_4_epoch_150.pt'
    model.load_state_dict(torch.load(reload_path))
    r = range(epoch+1,num_epochs)
for l in r:
    running_loss = 0.0
    avg_loss = 0.0
    print('epoch: ' + str(l))
    for k in range(int(len(x)/batch_size)):
        inputs, labels = next(data_gen)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        avg_loss += loss.item()

    model.eval()
    running_accuracy = 0
    for m in range(int(len(x_val)/batch_size)):
        inputs, labels = next(data_gen_val)
        outputs = model(inputs)
        labels_np = labels.cpu().numpy()
        outputs_np = outputs.cpu().detach().numpy()
        #print(labels_np)
        #print(outputs_np)
        accuracy = np.mean(np.argmax(outputs_np,1) == np.argmax(labels_np,1))
        running_accuracy += accuracy
        #print(accuracy)
    avg_accuracy = running_accuracy / int(len(x_val)/batch_size)
    print('avg validation accuracy: ' + str(avg_accuracy))

    running_accuracy = 0
    for m in range(int(len(x_test)/batch_size)):
        inputs, labels = next(data_gen_test)
        outputs = model(inputs)
        labels_np = labels.cpu().numpy()
        outputs_np = outputs.cpu().detach().numpy()
        #print(labels_np)
        #print(outputs_np)
        accuracy = np.mean(np.argmax(outputs_np,1) == np.argmax(labels_np,1))
        running_accuracy += accuracy
        #print(accuracy)
    avg_accuracy = running_accuracy / int(len(x_test)/batch_size)
    print('avg test accuracy: ' + str(avg_accuracy))

    model.train()

    state = model.state_dict()

    if l % 5 == 0:
        torch.save(state, 'trials/2D_'+subset+'_trainvaltest_retrain='+str(retrain)+'_lr_' + str(lr) + '_bs_' + str(batch_size) + '_epoch_' + str(l)+'.pt')
    print('avg loss this epoch: ' + str(avg_loss/(len(x)/batch_size)))
print('done')
