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
cluster = True
load_typedict = False
redo_tsne = True

id_to_im = pickle.load(open('storage/id_to_im.pkl', 'rb'))
ID_to_pred = pickle.load(open('storage/ID_to_pred.pkl', 'rb'))
total_var = np.load('clustering/total_var.npy')
within_var = np.load('clustering/within_var.npy')
features = np.load('clustering/features.npy')
IDs = pickle.load(open('clustering/IDs.pkl', 'rb'))
training = pickle.load(open('storage/x_train_inhibitory_excitatory.pkl', 'rb'))
training = [os.path.split(a)[-1] for a in training]

#plt.bar(range(within_var.size), within_var, color='green')
#plt.show()
#plt.bar(range(total_var.size), total_var, color='orange')
#plt.show()

synapse_file = 'Synapse_Table.xlsx'
naming_file = 'naming.xlsx'
cells_file = 'cells.xlsx'

df_synapse = pandas.read_excel(synapse_file)
df_name = pandas.read_excel(naming_file)
df_cell = pandas.read_excel(cells_file)

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
	
	for ID in IDs:
		row = df_synapse.loc[df_synapse['ID'] == ID]
		presyn = row.iloc[0]['Pre-Synaptic']
		if presyn in df_merged['Cell ID'].tolist():
			typerow = df_merged.loc[df_merged['Cell ID'] == presyn]
			cell_type = typerow.iloc[0]['Cell Type']
			typedict[ID] = cell_type.replace(' ','')
	pickle.dump(typedict, open('clustering/typedict.pkl', 'wb'))

if cluster:
	plt.figure(figsize=(25,15))
	excitatory_ims = os.listdir('Animal_1/excitatory')
	inhibitory_ims = os.listdir('Animal_1/inhibitory')
	if not redo_tsne and os.path.exists('clustering/tsne.npy'):
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
		ID = IDs[i]
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
	for i in range(len(IDs)):
		ID = IDs[i]
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
		train = any([a in training for a in imnames])
		if train:
			color = 'black'
		certainty = ID_to_pred[ID][1]
		if (gt in gts_used) or train:
			plt.scatter(X1, X2, color=color)
		else:
			plt.scatter(X1, X2, color=color, label=gt)
			gts_used.add(gt)
	plt.legend()
	plt.title('2D features for synapses w ground truth (0 = inhibitory, 1 = excitatory)')
	plt.savefig('clustering/figure3.png')
	plt.clf()
	#TODO: plot certainty (larger dots)

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

	print('done')


