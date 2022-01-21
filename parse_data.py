import os
from subprocess import call
import pandas
import math
import ipdb

imdir = 'Animal_1/cropped'
ntdir = 'Animal_1/'
#imdir = 'Animal_1/pr'
#ntdir = 'Animal_1/pr_sorted'
#imdir = 'Animal_1/rn_sorted/pr-amgrn'
#ntdir = 'Animal_1/rn_sorted/pr-amgrn'
csv_file = 'cells.xlsx'
name_file = 'naming.xlsx'

special_cells = ['ddn', 'mgin']
do_special_cells = False

df = pandas.read_excel(csv_file)
df_name = pandas.read_excel(name_file)

df['Cell ID'] = df['Cell ID'].astype(str).str.lower()
df_name['Final published name'] = df_name['Final published name'].astype(str).str.lower()
df_merged = df.merge(df_name, how='left', left_on='Cell ID', right_on='Final published name')

df_merged = df_merged.filter(['Cell ID','Cell Type','NT','Original Name','Alternate Name','Final published name'])

all_imnames = os.listdir(imdir)

df_name = df_name.applymap(str)
columns = df.columns
orig_names = pandas.Series(df_name['Original Name']).str.strip().str.lower()
df_name['Original Name'] = df_name['Original Name'].str.strip().str.lower()
df_name['Final published name'] = df_name['Final published name'].str.strip()


for index, row in df_merged.iterrows():
	cell_num = row[0]
	cell_id = row[1]
	NT = row[2]
	orig_name = row[3]
	alt_name = row[4]
	fin_name = row[5]

	names = [cell_num, cell_id, orig_name, alt_name, fin_name]
	names = [n for n in names if isinstance(n, str) or not math.isnan(n)]

	if (not NT) or (not isinstance(NT,str)) or ('?' in NT) or ('or' in NT):
		if not do_special_cells:
			continue

	if not do_special_cells and ((not isinstance(NT, str) and NT) or ('?' in NT) or ('or' in NT)):
		if cell_id in orig_names.unique():
			official_name = df_name.loc[df_name['Original Name'] == cell_id]['Final published name'].iloc[0]
			names.append(official_name)

	print(cell_id)
	print(NT)
	
	if do_special_cells:
		IDpath = os.path.join(ntdir, cell_id.lower().strip().replace(' ',''))
	else:
		NT = NT.lower().strip()
		nt_path = os.path.join(ntdir, NT)

	
	for name in names:
		for imname in all_imnames:
			imname_first = imname.split('_')[1]
			imname_syn = imname_first.split('-')[0].split('.')[0]
			if '00synapse' in imname_syn:
				imname_syn = imname_syn[9:]
			elif '00syn' in imname_syn:
				imname_syn = imname_syn[5:]
			if '}' in imname_syn:
				imname_syn_list = imname_syn.split('}')
				#print('name: ' + str(name))
				#print(imname_first)
				#print(imname_syn)
				#print(imname_syn_list)
			else:
				imname_syn_list = [imname_syn]

			#if 'coronet' in cell_id.lower():
			#	ipdb.set_trace()
			#prev_NT = None
			for imname_syn in imname_syn_list:
				if str(name).replace(' ','').lower() == imname_syn.lower():
					if do_special_cells:
						#if 'pr-amg' in cell_id.lower():
						#	ipdb.set_trace()
						if cell_id.lower().replace(' ','') in special_cells and os.path.exists(IDpath):
							if os.path.exists(os.path.join(IDpath, imname)):
								continue
							call(['scp', os.path.join(imdir, imname), os.path.join(IDpath, imname)])
					else:
						if os.path.exists(nt_path):
							copypath = os.path.join(nt_path, imname)
							if os.path.exists(copypath):
								continue
							call(['scp', os.path.join(imdir, imname), copypath])
						else:
							print('folder does not exist: ' + NT)
							continue

