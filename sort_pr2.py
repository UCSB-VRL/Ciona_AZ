import pandas
from subprocess import call
import os
import ipdb


orig_folder = '/media/angela/0d1ee072-3b3c-427f-ace6-c079402463b9/gdrive/Dataset1/Animal_1/pr_sorted/group2'
new_folder = os.path.join(orig_folder, 'sorted_postsynaptic')
csv_file = 'cells.xlsx'
df = pandas.read_excel(csv_file)
df.set_index('Cell ID', inplace=True)

for fn in os.listdir(orig_folder):
	if not fn.endswith('.tif'):
		continue
	f = fn.replace('}', '-')
	f = f.replace('_', '-')
	namelist = f.split('-')
	while ' ' in namelist:
		namelist.remove(' ')
	while '' in namelist:
		namelist.remove('')
	if len(namelist) < 5:
		ipdb.set_trace()
	presyn = namelist[2]
	if namelist[3] == 'pr':
		postsyn = 'pr' + namelist[4]
		fpath = os.path.join(orig_folder,fn)
		newpath = os.path.join(new_folder, 'pr2', fn)
		call(['scp', fpath, newpath])
	else:
		postsyn = namelist[3]
		if not postsyn.isnumeric():
			continue
		try:
			postrow = df.loc[int(postsyn)]
		except:
			print('key not found in df: ' + postsyn)
			continue
		if postrow[0].lower() == 'pr-amg rn':
			fpath = os.path.join(orig_folder,fn)
			newpath = os.path.join(new_folder, 'pr-amgrn', fn)
			call(['scp', fpath, newpath])
