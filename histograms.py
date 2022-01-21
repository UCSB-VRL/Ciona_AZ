import pandas
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import ipdb


cells_file = 'cells.xlsx'
df_cell = pandas.read_excel(cells_file)

cell_types = df_cell['Cell Type'].tolist()
cell_types = [t.replace(' ','').lower() for t in cell_types]
type_set = list(set(cell_types))
hist = []
for t in type_set:
	hist.append(cell_types.count(t))
x_pos = [i for i, _ in enumerate(type_set)]

plt.figure(figsize=(25,15))
plt.bar(x_pos, hist, color='green')
plt.xticks(x_pos, type_set, rotation='vertical')
plt.title('Number of Cells for each Cell Group (mean = ' +str(np.mean(hist)) + ')')

plt.savefig('stats/cells_per_group.png')
plt.clf()


################################
syn_path = 'Synapse_Predictions.csv'
synapse_df = pandas.read_csv(syn_path)

results = synapse_df[['Pre-Synaptic', 'Valence', 'Percentage', 'Certainty', 'Size']].dropna().groupby(['Pre-Synaptic', 'Valence']).agg({'Size':'count', 'Certainty':'mean'}).reset_index().pivot(index='Pre-Synaptic', columns='Valence').fillna(0)

#ipdb.set_trace()
colors_green = sns.light_palette("seagreen", as_cmap=True)
colors_red = sns.light_palette("tomato", as_cmap=True)
colors_list1 = []
colors_list2 = []
for i in range(colors_green.N):
	rgb1 = matplotlib.colors.rgb2hex(colors_green(i)[:3])
	colors_list1.append(rgb1)
	rgb2 = matplotlib.colors.rgb2hex(colors_red(i)[:3])
	colors_list2.append(rgb2)


names = results.index.tolist()
excitatory_bars = results[('Size', 'excitatory')].tolist()
inhibitory_bars = results[('Size', 'inhibitory')].tolist()
certainty_e = results[('Certainty', 'excitatory')].tolist()
certainty_i = results[('Certainty', 'inhibitory')].tolist()

old_min_e = min(certainty_e)
old_max_e = max(certainty_e)
old_min_i = min(certainty_i)
old_max_i = max(certainty_i)
new_max = 255
new_min = 0

new_cert_e = []
new_cert_i = []
for i in range(len(certainty_e)):
	old_value = certainty_e[i]
	new_value = ((old_value - old_min_e) / (old_max_e - old_min_e)) * (new_max - new_min) + new_min
	new_cert_e.append(colors_list1[int(np.floor(new_value))])
	old_value = certainty_i[i]
	new_value = ((old_value - old_min_i) / (old_max_i - old_min_i)) * (new_max - new_min) + new_min
	new_cert_i.append(colors_list2[int(np.floor(new_value))])

# set width of bar
barWidth = 0.4
 
# Set position of bar on X axis
r1 = np.arange(len(excitatory_bars))
r2 = [x + barWidth for x in r1]

'''
# Make the plot
plt.bar(r1, excitatory_bars, width=barWidth, color=new_cert_e, edgecolor='black', linewidth=.1)
plt.bar(r2, inhibitory_bars, width=barWidth, color=new_cert_i, edgecolor='black', linewidth=.1)
 
# Add xticks on the middle of the group bars
plt.xlabel('cell group (# = ' + str(len(names)) + ')', fontweight='bold')
plt.ylabel('# of synapses', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(names))], names, rotation='vertical')
plt.title('Inhibitory and Excitatory Predictions Per Cell (green/blue color = excitatory, red/yellow color = inhibitory, hue = certainty)')
# Create legend & Show graphic
#plt.legend()
plt.show()
'''

################################
results.index = results.index.str.lower()
results.columns = [' '.join(col).strip() for col in results.columns.values]
results = results.reset_index()
#results['Pre-Synaptic'] = results.index
df_cell['Cell ID'] = df_cell['Cell ID'].astype(str).str.replace(' ','').str.lower()
results_merged = df_cell.merge(results, left_on='Cell ID', right_on='Pre-Synaptic')
#ipdb.set_trace()
results_merged = results_merged[['Cell ID', 'Cell Type', 'NT', 'Size excitatory', 'Size inhibitory', 'Certainty excitatory', 'Certainty inhibitory']].groupby(['Cell Type', 'NT']).agg({'Cell ID':'count', 'Size excitatory':'sum', 'Size inhibitory':'sum', 'Certainty excitatory':'mean', 'Certainty inhibitory':'mean'}).reset_index()
results_merged = results_merged.rename({'Cell ID':'count'}, axis='columns')
results = results_merged[results_merged['Cell Type'].str.lower() != 'ambiguous']
results['NT'] = results['NT'].str.lower()

results = results.replace({'gly': 0})
results = results.replace({'gaba': 0})
results = results.replace({'glut': 1})
results = results.replace({'da': 1})
results = results.replace({'ach': 1})
#ipdb.set_trace()
### PLOT ###
colors_list1 = []
colors_list2 = []
for i in range(colors_green.N):
	rgb1 = matplotlib.colors.rgb2hex(colors_green(i)[:3])
	colors_list1.append(rgb1)
	rgb2 = matplotlib.colors.rgb2hex(colors_red(i)[:3])
	colors_list2.append(rgb2)


names = results['Cell Type'].tolist()
excitatory_bars = results['Size excitatory'].tolist()
inhibitory_bars = results['Size inhibitory'].tolist()
certainty_e = results['Certainty excitatory'].tolist()
certainty_i = results['Certainty inhibitory'].tolist()

old_min_e = min(certainty_e)
old_max_e = max(certainty_e)
old_min_i = min(certainty_i)
old_max_i = max(certainty_i)
new_max = 255
new_min = 0

new_cert_e = []
new_cert_i = []
for i in range(len(certainty_e)):
	old_value = certainty_e[i]
	new_value = ((old_value - old_min_e) / (old_max_e - old_min_e)) * (new_max - new_min) + new_min
	new_cert_e.append(colors_list1[int(np.floor(new_value))])
	old_value = certainty_i[i]
	new_value = ((old_value - old_min_i) / (old_max_i - old_min_i)) * (new_max - new_min) + new_min
	new_cert_i.append(colors_list2[int(np.floor(new_value))])

# set width of bar
barWidth = 0.4

# Set position of bar on X axis
r1 = np.arange(len(excitatory_bars))
r2 = [x + barWidth for x in r1]

plt.bar(r1, excitatory_bars, width=barWidth, color=new_cert_e, edgecolor='black', linewidth=.1)
plt.bar(r2, inhibitory_bars, width=barWidth, color=new_cert_i, edgecolor='black', linewidth=.1)


# Add xticks on the middle of the group bars
certainty_e = list(np.round(np.array(certainty_e) * 100))
for i, v in enumerate(certainty_e):
	plt.text(i-.15, 
		excitatory_bars[i] + 2, 
		int(certainty_e[i]), 
		fontsize=8, 
		color='green')

certainty_i = list(np.round(np.array(certainty_i) * 100))
# Add xticks on the middle of the group bars
for i, v in enumerate(certainty_i):
	plt.text(i+.25, 
		inhibitory_bars[i] + 2, 
		int(certainty_i[i]), 
		fontsize=8, 
		color='red')


plt.xlabel('cell group (# = ' + str(len(names)) + ')', fontweight='bold')
plt.ylabel('# of synapses', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(names))], names, rotation='vertical')
plt.title('Valence Predictions By Cell Type (green/blue color = excitatory, red/yellow color = inhibitory, hue/number on top = certainty)')
# Create legend & Show graphic
#plt.legend()
plt.show()

'''
#############################
#TODO: want split into excitatory and inhibitory by 0 and 1, drop all else
results = results[results['NT'].astype(str).str.contains('0|1')]
#ipdb.set_trace()
gt_e = results['NT'].tolist()
gt_i = results['NT'].tolist()
gt_e = [None if not isinstance(element, int) else element for element in gt_e]
gt_i = [2 if element==1 else element for element in gt_i]
gt_i = [None if not isinstance(element, int) else element for element in gt_i]
gt_i = [1 if (element==0) else element for element in gt_i]
gt_i = [0 if element==2 else element for element in gt_i]

names = results['Cell Type'].tolist()
excitatory_bars = results['Size excitatory'].tolist()
inhibitory_bars = results['Size inhibitory'].tolist()
certainty_e = results['Certainty excitatory'].tolist()
certainty_i = results['Certainty inhibitory'].tolist()

new_cert_e = []
new_cert_i = []
for i in range(len(certainty_e)):
	old_value = certainty_e[i]
	new_value = ((old_value - old_min_e) / (old_max_e - old_min_e)) * (new_max - new_min) + new_min
	new_cert_e.append(colors_list1[int(np.floor(new_value))])
	old_value = certainty_i[i]
	new_value = ((old_value - old_min_i) / (old_max_i - old_min_i)) * (new_max - new_min) + new_min
	new_cert_i.append(colors_list2[int(np.floor(new_value))])

# set width of bar
barWidth = 0.2

# Set position of bar on X axis
r1 = np.arange(len(excitatory_bars))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1, np.array(gt_i)*(np.array(excitatory_bars)+np.array(inhibitory_bars)), width=barWidth, color='red', hatch='x', edgecolor='black', linewidth=.1)
plt.bar(r2, inhibitory_bars, width=barWidth, color=new_cert_i, edgecolor='black', linewidth=.1)
plt.bar(r3, excitatory_bars, width=barWidth, color=new_cert_e, edgecolor='black', linewidth=.1)


# Add xticks on the middle of the group bars
plt.xlabel('cell group (# = ' + str(len(names)) + ')', fontweight='bold')
plt.ylabel('# of synapses', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(names))], names, rotation='vertical')
plt.title('Synapse Predictions and Ground Truth (Inhibitory, cross-hatched bar)')
# Create legend & Show graphic
#plt.legend()
plt.show()


plt.bar(r1, np.array(gt_e)*(np.array(excitatory_bars)+np.array(inhibitory_bars)), width=barWidth, color='green', hatch='x', edgecolor='black', linewidth=.1)
plt.bar(r2, excitatory_bars, width=barWidth, color=new_cert_e, edgecolor='black', linewidth=.1)
plt.bar(r3, inhibitory_bars, width=barWidth, color=new_cert_i, edgecolor='black', linewidth=.1)

# Add xticks on the middle of the group bars
plt.xlabel('cell group (# = ' + str(len(names)) + ')', fontweight='bold')
plt.ylabel('# of synapses', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(names))], names, rotation='vertical')
plt.title('Synapse Predictions and Ground Truth (Excitatory, cross-hatched bar)')
# Create legend & Show graphic
#plt.legend()
plt.show()
'''
