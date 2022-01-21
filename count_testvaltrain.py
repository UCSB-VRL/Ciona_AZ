import numpy as np
import pickle
import os
import ipdb


folder = 'storage'

x_train = pickle.load(open(os.path.join(folder, 'x_train_inhibitory_excitatory.pkl'), 'rb'))
x_val = pickle.load(open(os.path.join(folder, 'x_val_inhibitory_excitatory.pkl'), 'rb'))
x_test = pickle.load(open(os.path.join(folder, 'x_test_inhibitory_excitatory.pkl'), 'rb'))

#y_train = pickle.load(open(os.path.join(folder, 'y_train_inhibitory_excitatory.pkl'), 'rb'))
#y_val = pickle.load(open(os.path.join(folder, 'y_val_inhibitory_excitatory.pkl'), 'rb'))
#y_test = pickle.load(open(os.path.join(folder, 'y_test_inhibitory_excitatory.pkl'), 'rb'))

unique_cells_pre = {'train':set(), 'val':set(), 'test':set()}
excitatory = {'train':0, 'val':0, 'test':0}
inhibitory = {'train':0, 'val':0, 'test':0}

for x in x_train:
	if 'excitatory' in x:
		excitatory['train'] += 1
	elif 'inhibitory' in x:
		inhibitory['train'] += 1
	presyn = os.path.split(x)[-1].replace('_','-').split('-')[1].lower()
	if 'synapse' in presyn:
		presyn = presyn[9:]
	else:
		presyn = presyn[5:]
	unique_cells_pre['train'].add(presyn)

for x in x_val:
	if 'excitatory' in x:
		excitatory['val'] += 1
	elif 'inhibitory' in x:
		inhibitory['val'] += 1
	presyn = os.path.split(x)[-1].replace('_','-').split('-')[1].lower()
	if 'synapse' in presyn:
		presyn = presyn[9:]
	else:
		presyn = presyn[5:]
	unique_cells_pre['val'].add(presyn)

for x in x_test:
	if 'excitatory' in x:
		excitatory['test'] += 1
	elif 'inhibitory' in x:
		inhibitory['test'] += 1
	presyn = os.path.split(x)[-1].replace('_','-').split('-')[1].lower()
	if 'synapse' in presyn:
		presyn = presyn[9:]
	else:
		presyn = presyn[5:]
	unique_cells_pre['test'].add(presyn)

ipdb.set_trace()
