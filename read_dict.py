
import os
import pickle
import ipdb


folder = 'Dict'

syn_resolution = {}
resolutions = {'11500':564/2, '16500':267.0, '9900':319/2, '8200':492/2, '6000':354/2, '26500':375/0.5, '20500':290/0.5}
#resolutions = {'11500':105.5, '16500':72.0, '9900':120.0, '8200':78.0}
missing_res = set()

for d in os.listdir(folder):
	if d.endswith('.pkl'):
		print(d)
		dpath = os.path.join(folder, d)
		dictionary = pickle.load(open(dpath, 'rb'))
		if 'contours' in dictionary.keys():
			contours = dictionary['contours'].keys()
			for contour in contours:
				if ('00syn' in contour) or ('synapse' in contour):
					im_dict = dictionary['img']
					if 'name' not in im_dict:
						continue
					imname = im_dict['name']
					imname = imname.split('.')[0]
					resname = imname.split('-')[-1]
					resname = ''.join(c for c in resname if c.isdigit())
					if resname == '115002':
						resname = '11500'
					if resname not in resolutions.keys():
						print(resname)
						missing_res.add(resname)
						continue
					resolution = resolutions[resname]
					contour = contour.replace('/', '}')
					syn_resolution[contour] = resolution

print('total synapses with resolutions: ' + str(len(syn_resolution.keys())))
print('missing resolutions:')
print(missing_res)

pickle.dump(syn_resolution, open('storage/resolution_dict.pkl', 'wb'))
