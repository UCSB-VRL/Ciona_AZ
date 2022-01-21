import pandas
from subprocess import call
import os
import ipdb


orig_folder = '/media/angela/0d1ee072-3b3c-427f-ace6-c079402463b9/gdrive/Dataset1/Animal_1/rn_sorted/pr-amgrn'
new_folder = os.path.join(orig_folder, 'postsynaptic_sorted')
csv_file = 'cells.xlsx'
df = pandas.read_excel(csv_file)
df.set_index('Cell ID', inplace=True)

for fn in os.listdir(orig_folder):
	if not fn.endswith('.tif'):
		continue
	f = fn.replace('}', '-')
	f = f.replace('_', '-')
	f = f[:f.find('.tif')]
	namelist = f.split('-')
	while ' ' in namelist:
		namelist.remove(' ')
	while '' in namelist:
		namelist.remove('')
	if len(namelist) < 3:
		print('short name: ' + fn)
		continue
	postsyn = namelist[2]
	if not postsyn.isnumeric():
		continue
	try:
		postrow = df.loc[int(postsyn)]
	except:
		print('key not found in df: ' + postsyn)
		continue
	#print(postsyn)
	if postrow[0].lower() == 'pr-amg rn':
		#print('pr-amg rn')
		fpath = os.path.join(orig_folder,fn)
		newpath = os.path.join(new_folder, 'pr-amgrn', fn)
		call(['scp', fpath, newpath])
	elif postrow[0].lower() == 'prrn':
		#print('prrn')
		fpath = os.path.join(orig_folder,fn)
		newpath = os.path.join(new_folder, 'prrn', fn)
		call(['scp', fpath, newpath])
