import pandas
import numpy as np
import math
import os
import pickle
import ipdb

combine = True
if combine:
    id_to_im_1 = pickle.load(open('storage/id_to_im.pkl', 'rb'))
    id_to_im_2 = pickle.load(open('storage_copy_1/id_to_im.pkl', 'rb'))
    print('original size: ' + str(len(id_to_im_1.keys())) + ' ' + str(len(id_to_im_2.keys())))
    id_to_im_1.update(id_to_im_2)
    print('new size: ' + str(len(id_to_im_1.keys())))
    pickle.dump(id_to_im_1, open('storage/id_to_im.pkl', 'wb'))
    quit()

do_underscore = False
        
synapse_file = 'Synapse_Table.csv'
naming_file = 'naming.csv'
cells_file = 'cells.csv'
im_folder = 'Animal_1/cropped'

df_synapse = pandas.read_csv(synapse_file)
df_name = pandas.read_csv(naming_file)
df_cell = pandas.read_csv(cells_file)
id_to_im = {}

all_names = os.listdir(im_folder)

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

synapse_pre = df_synapse['Pre-Synaptic'].tolist()
synapse_post = df_synapse['Post-Synaptic'].tolist()
ids = df_cell['Cell ID'].tolist()
final_names = df_name['Final published name'].tolist()
alt_names = df_name['Alternate Name'].tolist()
orig_names = df_name['Original Name'].tolist()

final_names_merged = df_merged['Final published name'].tolist()
alt_names_merged = df_merged['Alternate Name'].tolist()
orig_names_merged = df_merged['Original Name'].tolist()

not_found_in_table = set()
pre_to_im = {}
for name in all_names:
    if name.startswith('._'):
        name = name[2:]
    synapse_id = '_'.join(name.split('_')[1:])
    if 'pr' in name.split('-')[0]:
        #ipdb.set_trace()
        new_name = name[name.find('pr'):name.find('pr')+4]
        #if len(name.split('-')[1]) == 1:
        #   ipdb.set_trace()
        new_name = ''.join(ch for ch in new_name if ch.isalnum())
        old_name = new_name
        for c in range(len(old_name)):
            char = old_name[c]
            if len(old_name) > 3 and old_name[2].isnumeric() and c > 2:
                if not char.isnumeric():
                    del new_name[c]
        if len(new_name) == 3 and not new_name[2].isnumeric():
            new_name = 'pr-' + new_name[2]
    else:
        if 'synapse' in name:
            new_name = name[name.find('00synapse'):name.find('-')][9:]
        else:
            new_name = name[name.find('00syn'):name.find('-')][5:]
    #if '95' in new_name:
    #   ipdb.set_trace()
    
    #TODO: exclude dyads where they are of different NT?
    f = name.replace('}', '-')
    if do_underscore:
        underscore_inds = find_all(f, '_')
        flist = f.split('_')
        fnum = flist[0]
        fname = '_'.join(flist[1:])
        while '_' in fname:
            print(fname)
            firstind = fname.find('_')
            substr = fname[0:firstind]
            nextind = fname[firstind:].find('-')
            if nextind > 0:
                nextind += firstind
            fname = substr + fname[nextind:]
        f = '_'.join([fnum,fname])
    else:
        f = f.replace('_', '-')
        
    namelist = f.split('-')
    while ' ' in namelist:
        namelist.remove(' ')
    while '' in namelist:
        namelist.remove('')
    if len(namelist) < 3:
        continue
    if 'pr' in name.split('-')[0]:
        #ipdb.set_trace()
        if len(namelist) < 3:
            continue
        prind = 2
        if '-' in new_name:
            if len(namelist) < 4:
                continue
            prind = 3
        postsyn = namelist[prind]
        if postsyn == 'pr':
            postsyn = '-'.join([postsyn, namelist[prind+1]])
    else:
        postsyn = namelist[2]
        if postsyn == 'pr':
            if len(namelist) < 4:
                continue
            postsyn = '-'.join([postsyn, namelist[3]])

    postsyn = postsyn.split('.')[0]
    new_name = new_name.lower()
    postsyn = postsyn.lower()
    possible_names = []
    possible_postsyn = []

    if new_name in orig_names:
        if new_name in orig_names_merged:
            name_row = df_merged.loc[df_merged['Original Name'] == new_name]
        else:
            possible_names.append(new_name)
    elif new_name in alt_names:
        if new_name in alt_names_merged:
            name_row = df_merged.loc[df_merged['Alternate Name'] == new_name]
        else:
            possible_names.append(new_name)
    elif new_name in final_names:
        if new_name in final_names_merged:
            name_row = df_merged.loc[df_merged['Final published name'] == new_name]
        else:
            possible_names.append(new_name)
    elif new_name in ids:
        name_row = df_merged.loc[df_merged['Cell ID'] == new_name]
    else:
        #print('name not found: ' + new_name)
        possible_names = [new_name]
        continue

    if len(possible_names) == 0:
        possible_names = [name_row.iloc[0]['Original Name'], name_row.iloc[0]['Alternate Name'], name_row.iloc[0]['Final published name'], name_row.iloc[0]['Cell ID']]
        possible_names = [n for n in possible_names if ((isinstance(n, str) or not math.isnan(n)) and n!='nan')]


    if postsyn in orig_names:
        if postsyn in orig_names_merged:
            postsyn_row = df_merged.loc[df_merged['Original Name'] == postsyn]
        else:
            possible_postsyn.append(postsyn)
    elif postsyn in alt_names:
        if postsyn in alt_names_merged:
            postsyn_row = df_merged.loc[df_merged['Alternate Name'] == postsyn]
        else:
            possible_postsyn.append(postsyn)
    elif postsyn in final_names:
        if postsyn in final_names_merged:
            postsyn_row = df_merged.loc[df_merged['Final published name'] == postsyn]
        else:
            possible_postsyn.append(postsyn)
    elif postsyn in ids:
        postsyn_row = df_merged.loc[df_merged['Cell ID'] == postsyn]
    else:
        possible_postsyn = [postsyn]
        #print('postsyn not found: ' + postsyn)
        #continue

    if len(possible_postsyn) == 0:
        possible_postsyn = [postsyn_row.iloc[0]['Original Name'], postsyn_row.iloc[0]['Alternate Name'], postsyn_row.iloc[0]['Final published name'], postsyn_row.iloc[0]['Cell ID']]
        possible_postsyn = [n for n in possible_postsyn if ((isinstance(n, str) or not math.isnan(n)) and n!='nan')]

    found = False
    for pre in possible_names:
        for post in possible_postsyn:
            #if pre not in synapse_pre or post not in synapse_post:
                #print('not in synapse lists: ' + pre + ' or ' + post)
            #   continue
            row = df_synapse.loc[(df_synapse['Pre-Synaptic']==pre) & (df_synapse['Post-Synaptic']==post)]
            if row.empty:
                continue
            found = True
            ID = row.iloc[0]['ID']
            if ID in id_to_im.keys():
                id_to_im[ID].add(name)
            else:
                id_to_im[ID] = set()
                id_to_im[ID].add(name)
        pre_row = df_synapse.loc[(df_synapse['Pre-Synaptic']==pre)]
        if pre_row.empty or found:
            continue
        syn_name = '_'.join(name.split('_')[1:])
        if pre in pre_to_im.keys():
            if syn_name in pre_to_im[pre].keys():
                pre_to_im[pre][syn_name].add(name)
            else:
                pre_to_im[pre][syn_name] = set()
                pre_to_im[pre][syn_name].add(name)
        else:
            pre_to_im[pre] = {syn_name: set()}
            pre_to_im[pre][syn_name].add(name)
    if not found:
        #ipdb.set_trace()
        not_found_in_table.add(((pre, post), name))
        
not_found_in_table = list(not_found_in_table)
not_found_in_table = [list(x[0])+[x[1]] for x in not_found_in_table]

pickle.dump(id_to_im, open('storage/id_to_im.pkl', 'wb'))
print('complete: found ' + str(len(id_to_im.keys())) + ' IDs')

pickle.dump(not_found_in_table, open('storage/not_found.pkl', 'wb'))
print('not found: ' + str(len(not_found_in_table)) + ' IDs')

pickle.dump(pre_to_im, open('storage/pre_to_im.pkl', 'wb'))

not_found_df = pandas.DataFrame(list(not_found_in_table), columns =['Pre', 'Post', 'Name'])
not_found_df.to_csv('annotations_no_match.csv', index=False)
