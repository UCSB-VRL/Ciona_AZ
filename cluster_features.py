from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import pandas
import itertools
import os
import pickle
import ipdb


N = 500
do_postsynaptic = True
plot_only = True 
cluster = True
load_typedict = True
redo_tsne = True

id_to_im = pickle.load(open('storage/id_to_im.pkl', 'rb'))

if plot_only:
    total_var = np.load('clustering/total_var.npy')
    within_var = np.load('clustering/within_var.npy')
    features = np.load('clustering/features.npy')
    IDs = pickle.load(open('clustering/IDs.pkl', 'rb'))
    im_to_cell = pickle.load(open('storage/im_to_cell.pkl', 'rb'))

    #plt.bar(range(within_var.size), within_var, color='green')
    #plt.show()
    #plt.bar(range(total_var.size), total_var, color='orange')
    #plt.show()

    synapse_file = 'Synapse_Table.csv'
    naming_file = 'naming.csv'
    cells_file = 'cells.csv'

    df_synapse = pandas.read_csv(synapse_file)
    df_name = pandas.read_csv(naming_file)
    df_cell = pandas.read_csv(cells_file)

    df_cell['Cell ID'] = df_cell['Cell ID'].astype(str).str.lower()
    df_name['Final published name'] = df_name['Final published name'].astype(str).str.lower()
    df_merged = df_cell.merge(df_name, how='left', left_on='Cell ID', right_on='Final published name')
    df_merged['Final published name'] = df_merged['Final published name'].astype(str).str.lower()
    df_merged['Alternate Name'] = df_merged['Alternate Name'].astype(str).str.lower()
    df_merged['Original Name'] = df_merged['Original Name'].astype(str).str.lower()
    df_synapse['Pre-Synaptic'] = df_synapse['Pre-Synaptic'].astype(str).str.lower()
    df_synapse['Post-Synaptic'] = df_synapse['Post-Synaptic'].astype(str).str.lower()
    df_name['Final published name'] = df_name['Final published name'].astype(str).str.lower()
    df_name['Alternate Name'] = df_name['Alternate Name'].astype(str).str.lower()
    df_name['Original Name'] = df_name['Original Name'].astype(str).str.lower()

    if load_typedict:
        typedict = pickle.load(open('clustering/typedict.pkl', 'rb'))
    else:
        typedict = {}
        
        all_presyn = set()
        for ID in IDs:
            row = df_synapse.loc[df_synapse['ID'] == ID]
            presyn = row.iloc[0]['Pre-Synaptic']
            all_presyn.add(presyn)
            if presyn in df_merged['Cell ID'].tolist():
                typerow = df_merged.loc[df_merged['Cell ID'] == presyn]
                cell_type = typerow.iloc[0]['Cell Type']
                typedict[ID] = cell_type.replace(' ','')
        pickle.dump(typedict, open('clustering/typedict.pkl', 'wb'))


    if cluster:
        plt.figure(figsize=(25,15))
        excitatory_ims = os.listdir('Animal_1/excitatory')
        inhibitory_ims = os.listdir('Animal_1/inhibitory')
        gaba_ims = os.listdir('Animal_1/gaba')
        glut_ims = os.listdir('Animal_1/glut')
        gly_ims = os.listdir('Animal_1/gly')
        ach_ims = os.listdir('Animal_1/ach')
        if (not redo_tsne) and os.path.exists('clustering/tsne.npy'):
            X_embedded = np.load('clustering/tsne.npy')
        else:
            pca = PCA(n_components=50)
            #pca = PCA(n_components=2)
            pca.fit(features)
            reduced = pca.transform(features)
            X_embedded = TSNE(n_components=2, init='pca', random_state=1).fit_transform(reduced)
            #X_embedded = reduced
            np.save('clustering/reduced_pca.npy', reduced)
            np.save('clustering/tsne.npy', X_embedded)
        del features
        df_merged['Cell Type'] = df_merged['Cell Type'].str.replace(' ', '')
        unique_groups = list(set(df_merged['Cell Type'].tolist()))
        palette = sns.color_palette("Set2").as_hex()
        #ipdb.set_trace()
        markers = ["o","^","s","P", "*"]
        color_marker = list(set(list(itertools.product(palette, markers))))
        #ipdb.set_trace()
        group_to_marker = {}
        for g in range(len(unique_groups)):
            group = unique_groups[g]
            if group in group_to_marker:
                ipdb.set_trace()
            group_to_marker[group] = color_marker[g]
            #print(color_marker[g])
        group_loc_dict = {}
        FE = 0
        TE = 0
        FI = 0
        TI = 0
        for i in range(len(IDs)):
            #ipdb.set_trace()
            ID = IDs[i]
            if ID not in typedict.keys():
                continue
            imnames = list(id_to_im[ID])
            gt = None
            if imnames[0] in inhibitory_ims:
                #ipdb.set_trace()
                gt = 0
            elif imnames[0] in excitatory_ims:
                gt = 1
            X1 = X_embedded[i,0]
            X2 = X_embedded[i,1]
            #ipdb.set_trace()
            if X1 < 0:
                pred = 1
            else:
                pred = 0
            if gt is not None:
                if pred == gt:
                    if pred == 1:
                        TE += 1
                    else:
                        TI += 1
                else:
                    if pred == 1:
                        FE += 1
                    else:
                        FI += 1
            group = typedict[ID].replace(' ','')
            color, marker = group_to_marker[group]
            #ipdb.set_trace()
            if group in group_loc_dict.keys():
                plt.scatter(X1, X2, color=color, marker=marker)
                group_loc_dict[group].append([X1, X2])
            else:
                plt.scatter(X1, X2, color=color, marker=marker, label=group)
                group_loc_dict[group] = [[X1, X2]]

        plt.legend(bbox_to_anchor=(1.11, 1.06))
        plt.title('t-distributed Stochastic Neighbor Embedding for ' + str(len(group_loc_dict.keys())) + ' groups after PCA')
        plt.savefig('clustering/figure1.png')
        plt.clf()

        # --------- Plot average locations ---------- #
        for group in group_loc_dict.keys():
            coords = group_loc_dict[group]
            mean_coords = np.mean(coords, axis=0)
            color, marker = group_to_marker[group]
            plt.scatter(mean_coords[0], mean_coords[1], color=color, marker=marker, s=400, label=group)
        plt.legend(handleheight=2.5, bbox_to_anchor=(1.11, 1.06))
        plt.title('group average coords in feature space after PCA and tSNE')
        plt.savefig('clustering/figure2.png')
        plt.clf()

        # --------- Plot by GT ---------- #
        gts_used = set()
        labels = ['inhibitory', 'excitatory']
        for i in range(len(IDs)):
            ID = IDs[i]
            if ID not in typedict.keys():
                continue
            imnames = list(id_to_im[ID])
            gt = None
            if imnames[0] in inhibitory_ims:
                #ipdb.set_trace()
                gt = 0
            elif imnames[0] in excitatory_ims:
                gt = 1
            if gt is None:
                continue
            X1 = X_embedded[i,0]
            X2 = X_embedded[i,1]
            group = typedict[ID].replace(' ','')
            color = palette[gt]
            if gt in gts_used:
                plt.scatter(X1, X2, color=color)
            else:
                plt.scatter(X1, X2, color=color, label=labels[gt])
                gts_used.add(gt)
        plt.legend()
        plt.title('2D features for synapses w ground truth (0 = inhibitory, 1= excitatory)')
        plt.savefig('clustering/figure3.png')
        plt.clf()

        # --------- Plot by NT ---------- #
        gts_used = set()
        labels = ['gaba', 'gly', 'glut', 'ach']
        for i in range(len(IDs)):
            ID = IDs[i]
            if ID not in typedict.keys():
                continue
            imnames = list(id_to_im[ID])
            gt = None
            if imnames[0] in gaba_ims:
                #ipdb.set_trace()
                gt = 0
            elif imnames[0] in gly_ims:
                gt = 1
            elif imnames[0] in glut_ims:
                gt = 2
            elif imnames[0] in ach_ims:
                gt = 3
            if gt is None:
                continue
            X1 = X_embedded[i,0]
            X2 = X_embedded[i,1]
            group = typedict[ID].replace(' ','')
            color = palette[gt]
            if gt in gts_used:
                plt.scatter(X1, X2, color=color)
            else:
                plt.scatter(X1, X2, color=color, label=labels[gt])
                gts_used.add(gt)
        plt.legend()
        plt.title('2D features for synapses w ground truth (0 = inhibitory, 1= excitatory)')
        plt.savefig('clustering/figure4.png')
        plt.clf()
        
        # --------- Plot by Cell Type ---------- #
        cts_used = set()
        #labels = ['PR(I)', 'PR(II)', 'MGIN', 'AntRN', 'pr-AMGRN', 'pr-BTNRN', 'PNRN']
        labels = ['MGIN', 'AntRN', 'pr-AMGRN', 'pr-BTNRN', 'PNRN', 'prRN', 'pr-corRN']
        for i in range(len(IDs)):
            ID = IDs[i]
            if ID not in typedict.keys():
                continue
            group = typedict[ID].replace(' ','')
            if group not in labels:
                continue
            print(group)
            color, marker = group_to_marker[group]
            X1 = X_embedded[i,0]
            X2 = X_embedded[i,1]
            if group in cts_used:
                plt.scatter(X1, X2, color=color, marker=marker)
            else:
                plt.scatter(X1, X2, color=color, marker=marker, label=group)
                cts_used.add(group)
        plt.legend()
        plt.title('2D features for pr and rns')
        plt.savefig('clustering/figure5.png')
        plt.clf()
        

        # --------- Plot CM ---------- #
        cm = [[TE, FE], [FI, TI]]
        print(cm)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='g', annot_kws={"size": 14})
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.title('confusion matrix for clustering (0 = inhibitory, 1 = excitatory)')
        plt.savefig('clustering/cm.png')
        plt.clf()

        print('done clustering')
    

else:
    import torch
    from PIL import Image
    from skimage.transform import resize
    import gc
    import GPUtil

    def preprocess(batch, mean, std):
        batch -= mean
        batch /= std
        return batch

    image_folder = 'Animal_1/cropped'
    all_folders = [image_folder]
    num_classes = 2
    classes = ['inhibitory', 'excitatory']
    model = torch.load('../trials/resnext_pretrained.pt')
    subset = 'ach_gaba'
    reload_stats = True
    ID_pred_path = 'storage/ID_to_pred.pkl'
    
    ID_to_pred={}

    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Linear(2048, num_classes, bias=False)

    reload_path = 'trials/2D_inhibitory_excitatory_trainvaltest_retrain=True_lr_0.0001_bs_2_epoch_200.pt'
    model.load_state_dict(torch.load(reload_path))


    class Identity(torch.nn.Module):
        def __init__(self):
            super(Identity, self).__init__()
            
        def forward(self, x):
            return x

    model.fc = Identity()
    #model.avgpool = Identity()

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

    total_batch_size = 32

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
                x.remove(im_name)
                continue
            elif len(img.shape) > 2:
                img = img[:,:,0]
            #img = resize(img, output_shape = [N,N], preserve_range=True, mode = 'constant', order = 1)
            stats_stack[0] += np.mean(img)
            stats_stack[1] += np.std(img)

        total_mean = stats_stack[0]/len(x)
        total_std = stats_stack[1]/len(x)
        print('mean: ' + str(total_mean))
        print('std: ' + str(total_std))
        np.save('storage/mean.npy', total_mean)
        np.save('storage/std.npy', total_std)

    id_keys = list(id_to_im.keys())

    N_train = len(x)
    del x

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


    data_gen = ciona_data_gen()

    features = []
    IDs = []
    within_var = []
    total_var = []

    with torch.no_grad():
        model.eval()
        ''' Prediction '''
        for k in range(len(id_keys)):
            ID, names, inputs = next(data_gen)
            if ID in ID_to_pred.keys():
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
            features.append(np.mean(outputs_np, axis=0))
            IDs.append(ID)
            within_var.append(np.std(outputs_np, axis=0))
            for out in range(outputs_np.shape[0]):
                total_var.append(outputs_np[out])
            del inputs
            del names
            del ID
            del total_outputs
            del outputs_np
            #ipdb.set_trace()
        total_var = np.array(total_var)
        within_var = np.array(within_var)
        total_var = np.std(total_var, axis=0)
        within_var = np.mean(within_var, axis=0)
        np.save('clustering/total_var.npy', total_var)
        np.save('clustering/within_var.npy', within_var)
        np.save('clustering/features.npy', features)
        pickle.dump(IDs, open('clustering/IDs.pkl', 'wb'))
        print('done with pred')



