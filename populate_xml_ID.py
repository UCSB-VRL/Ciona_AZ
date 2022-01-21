import numpy as np
import pandas
import pickle
import os
import ipdb

plot_cm = False

syn_path = 'Synapse_Table.csv'
synapse_df = pandas.read_csv(syn_path)
ID_to_pred = pickle.load(open('storage/ID_to_pred.pkl', 'rb'))
categories = ['inhibitory', 'excitatory']
#ategories = ['ach', 'gaba', 'glut', 'gly']
x_train = pickle.load(open('storage/x_train_ach_gaba_gly_glut.pkl', 'rb'))
id_to_im = pickle.load(open('storage/id_to_im.pkl', 'rb'))
pre_to_pred = pickle.load(open('storage/pre_to_pred.pkl', 'rb'))
im_to_synapse = pickle.load(open('storage/im_to_synapse.pkl','rb'))
im_to_cell = pickle.load(open('storage/im_to_cell.pkl','rb'))
train_cells = pickle.load(open('storage/train_cells.pkl','rb'))
test_cells = pickle.load(open('storage/test_cells.pkl','rb'))
#overlap_F = open('storage/overlap.txt', 'r')
#overlap_list = overlap_F.read().splitlines()
#overlap_F.close()
overlap_list = []

x_train = [os.path.split(x)[-1] for x in x_train]
#ipdb.set_trace()

if plot_cm:
    import seaborn as sns
    from matplotlib import pyplot as plt
    id_to_im = pickle.load(open('storage/id_to_im.pkl', 'rb'))
    i_ims = os.listdir('Animal_1/inhibitory')
    e_ims = os.listdir('Animal_1/excitatory')
    TE = 0
    TI = 0
    FE = 0
    FI = 0
    
#train_cells = set()
#test_cells = set()

for key in ID_to_pred.keys():
    train = False
    print(key)
    val = ID_to_pred[key]
    im = list(id_to_im[key])[0]
    if im not in im_to_cell.keys() or im in overlap_list:
        cell = None
    else:
        cell = im_to_cell[im]
        print(cell)
    if im in x_train:
        train = True
        #if cell in test_cells and cell is not None:
        #    print('cell in test but now train')
        #    ipdb.set_trace()
        #train_cells.add(cell)
   # else:
        #if cell in train_cells and cell is not None:
         #   print('cell in train but now test')
         #   ipdb.set_trace()
        #test_cells.add(cell)
        
    if plot_cm:
        pred = int(np.round(val[0]))
        gt = None
        ims = list(id_to_im[key])
        if ims[0] in i_ims:
            gt = 0
        elif ims[0] in e_ims:
            gt = 1
        if gt is None:
            continue
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
    else:
        pred = categories[int(np.round(val[0]))]
        certainty = val[1]
        '''
        if val[0] > 0.5:
            certainty = val[0]
        elif val[0] < 0.5:
            certainty = 1-val[0]
        else:
            certainty = 0.5
        '''
        ID_to_pred[key] = [pred, certainty, val[2], train]

if plot_cm:
    cm = [[TE, FE], [FI, TI]]
    print(cm)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='g', annot_kws={"size": 14})
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('confusion matrix for CNN (0 = inhibitory, 1 = excitatory)')
    plt.savefig('clustering/cm_nn.png')
    plt.clf()
else:
    id_df = pandas.DataFrame.from_dict(ID_to_pred, orient='index')
    id_df.reset_index(level=0,inplace=True)
    #ipdb.set_trace()
    id_df.columns = ['ID', 'NT', 'Certainty', 'Size', 'Training']
    id_df['Certainty'] /= id_df['Certainty'].max()
    merged_df = pandas.merge(synapse_df, id_df, how='left', on='ID')

    merged_df.to_csv('Synapse_Predictions.csv', index=False)
print('done with IDs')


new_pre_to_pred = {}
#ipdb.set_trace()
for pre in pre_to_pred.keys():
    val = pre_to_pred[pre]
    #print(val)
    #ipdb.set_trace()
    estimate, certainty, num_synapses, num_ims = val[0]
    pred = categories[int(np.round(estimate))]
    '''
    if estimate > 0.5:
        estimate = estimate
    elif estimate < 0.5:
        estimate = 1-estimate
    else:
        estimate = 0.5
    '''
    new_pre_to_pred[pre] = [pred, certainty, num_synapses, num_ims]
pre_df = pandas.DataFrame.from_dict(new_pre_to_pred, orient='index')
pre_df.reset_index(level=0, inplace=True)
pre_df.columns = ['presynaptic cell', 'valence', 'certainty', 'num_synapses', 'num_ims']
pre_df.to_csv('Presynaptic Cell Predictions.csv', index=False)
