import numpy as np
import cv2
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import XMLParser
import pickle
import os
import re
import string
import ipdb


def dict_compare(d1, d2):
	d1_keys = set(d1.keys())
	d2_keys = set(d2.keys())
	shared_keys = d1_keys.intersection(d2_keys)
	added = d1_keys - d2_keys
	removed = d2_keys - d1_keys
	modified = {o : (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
	same = set(o for o in shared_keys if d1[o] == d2[o])
	return added, removed, modified, same


def clear_comments(path):
	# Using readline() 
	file0 = open(os.path.join('No_Comments', os.path.split(path)[-1]), 'w')  
	file1 = open(path, 'r') 
	count = 0
	lines_to_write = []
	prev_lines_to_write = []
	prev_line = ''
	prev_prev_line = ''
	skip = False
	redo_prev_line = False
	while True: 
		count += 1
		# Get next line from file 
		try:
			line = file1.readline()
		except:
			print("Line{}: {}".format(count, line.strip()))
			skip = True
			#ipdb.set_trace()
			continue
		valid_char = string.printable
		valid_char = valid_char.replace('\x0b','')
		valid_char = valid_char.replace('\x0c','')
		#line = re.sub(r'[^\x00-\x7f]',r'', line)
		line = ''.join(i for i in line if i in valid_char)
		#if 'comment' in line:
		#	skip = True
		old_line = line
		line_list = line.split(' ')
		for i in range(len(line_list)):
			if 'comment' in line_list[i]:
				line_list[i] = ''
			if 'comment' in old_line and '=' not in line_list[i]:
				line_list[i] = ''
		line = ' '.join(line_list)
		if line.strip().startswith('<'):
			if not skip:
				if (len(lines_to_write) > 0) and (prev_lines_to_write == lines_to_write):
					if lines_to_write[0].strip().startswith('</'):
						lines_to_write = []
				if len(lines_to_write) > 0 and len(prev_lines_to_write) > 0:
					if lines_to_write[0].strip().startswith('<Contour') and prev_lines_to_write[0].strip().startswith('</Transform'):
						lines_to_write = []
						redo_prev_line = True
				if redo_prev_line:
					redo_prev_line = False
				else:
					prev_lines_to_write = lines_to_write
				file0.writelines(lines_to_write)

			lines_to_write = []
		
		lines_to_write.append(line)
		if not line:
			if not skip:
				file0.writelines(lines_to_write) 
			break
		if '1.458' in path and count==2987:
			print("Line{}: {}".format(count, line.strip()))
			#ipdb.set_trace()
		skip = False
	file0.close() 
	file1.close() 


def parse_xml(path):
	def parse_coef(str):
		split_str = str.split(' ')
		coef = [float(i) for i in split_str if i]
		return coef

	def parse_points(str):
		split_str = str.split(',  ')
		points = []
		for s in split_str:
			if s:
				double_split_str = s.split(' ')
				points.append([float(i) for i in double_split_str])
		return points
	
	xml_dict = {'img': {}, 'contours': {}}
	parser = XMLParser()
	#parser = XMLParser(encoding='ISO-8859-15')
	tree = ET.parse(path, parser=parser)
	root = tree.getroot()
	for child in root:
		for gchild in child:
			if gchild.tag == 'Image':
				xml_dict['mag'] = float(gchild.attrib['mag'])
				xml_dict['img']['name'] = gchild.attrib['src']
				xml_dict['img']['xcoef'] = parse_coef(child.attrib['xcoef'])
				xml_dict['img']['ycoef'] = parse_coef(child.attrib['ycoef'])
				xml_dict['img']['dim'] = float(child.attrib['dim'])
			elif gchild.tag == 'Contour':
				contour_name = gchild.attrib['name']
				xml_dict['contours'][contour_name] = {}
				xml_dict['contours'][contour_name]['points'] = parse_points(gchild.attrib['points'])
				xml_dict['contours'][contour_name]['xcoef'] = parse_coef(child.attrib['xcoef'])
				xml_dict['contours'][contour_name]['ycoef'] = parse_coef(child.attrib['ycoef'])
				xml_dict['contours'][contour_name]['dim'] = float(child.attrib['dim'])
	return xml_dict


def Xforward(dim, a, b, x, y):
	if dim == 1:
		return a[0] + x
	elif dim == 2:
		return a[0] + a[1]*x
	elif dim == 3:
		return a[0] + a[1]*x + a[2]*y
	elif dim == 4:
		return a[0] + (a[1] + a[3]*y)*x + a[2]*y
	elif dim == 5:
		return a[0] + (a[1] + a[3]*y + a[4]*x)*x + a[2]*y
	elif dim == 6:
		return a[0] + (a[1] + a[3]*y + a[4]*x)*x + (a[2] + a[5]*y)*y
	return None


def Yforward(dim, a, b, x, y):
	if dim == 1:
		return b[0] + y
	elif dim == 2:
		return b[0] + b[1]*y
	elif dim == 3:
		return b[0] + b[1]*x + b[2]*y
	elif dim == 4:
		return b[0] + (b[1] + b[3]*y)*x + b[2]*y
	elif dim == 5:
		return b[0] + (b[1] + b[3]*y + b[4]*x)*x + b[2]*y
	elif dim == 6:
		return b[0] + (b[1] + b[3]*y + b[4]*x)*x + (b[2] + b[5]*y)*y
	return None


def XYinverse(dim, a, b, x, y):
	epsilon = 5e-10
	if dim == 0:
		return (x,y)
	elif dim == 1:
		x = x - a[0]
		y = y - b[0]
	elif (dim == 2) or (dim == 3):
		u = x - a[0]
		v = y - b[0]
		p = a[1]*b[2] - a[2]*b[1]
		if abs(p) > epsilon:
			x = (b[2]*u - a[2]*v)/p
			y = (a[1]*v - b[1]*u)/p
	elif (dim == 4) or (dim == 5) or (dim == 6):
		u = x
		v = y
		x0 = 0.0		   #initial guess of (x,y)
		y0 = 0.0
		u0 = Xforward(dim, a, b, x0,y0)		  #get forward tform of initial guess
		v0 = Yforward(dim, a, b, x0,y0)
		i = 0
		e = 1.0
		while (e > epsilon) and (i < 10):
			i += 1
			l = a[1] + a[3]*y0 + 2.0*a[4]*x0
			m = a[2] + a[3]*x0 + 2.0*a[5]*y0
			n = b[1] + b[3]*y0 + 2.0*b[4]*x0
			o = b[2] + b[3]*x0 + 2.0*b[5]*y0
			p = l*o - m*n
			if abs(p) > epsilon:
				x0 += (o*(u-u0) - m*(v-v0))/p
				y0 += (l*(v-v0) - n*(u-u0))/p
			else:
				x0 += l*(u-u0) + n*(v-v0)
				y0 += m*(u-u0) + o*(v-v0)
			u0 = Xforward(dim, a, b, x0,y0)
			v0 = Yforward(dim, a, b, x0,y0)
			e = abs(u-u0) + abs(v-v0)
		x = x0
		y = y0
	return (x,y)


#path = 'Series1.7661.7661'
#path = 'Series1.185'
#path = 'Series1.505'
#path = 'Series1.185'
#path = 'Series1.187'
cont_name = 'q3cb'

series_files = os.listdir('Series')
series_files.sort()
for series in series_files:
	print(series)
	if not series.startswith('Series'):
		continue
	path = os.path.join('Series', series)
	new_path = os.path.join('No_Comments', series)
	clear_comments(path)
	xml_dict = parse_xml(new_path)
	dict_file = os.path.join('Dict', series+'.pkl')
	with open(dict_file, 'wb') as f:
		pickle.dump(xml_dict, f)

#xml_dict = parse_xml(path)
#y_dict = parse_xml(path+'_no_comments')
#added, removed, modified, same = dict_compare(xml_dict['contours'], y_dict['contours'])

ipdb.set_trace()

src = xml_dict['img']['name']
a_img = xml_dict['img']['xcoef']
b_img = xml_dict['img']['ycoef']
dim_img = xml_dict['img']['dim']
mag = xml_dict['mag']

a_cont = xml_dict['contours'][cont_name]['xcoef']
b_cont = xml_dict['contours'][cont_name]['ycoef']
cont_points = xml_dict['contours'][cont_name]['points']
dim_cont = xml_dict['contours'][cont_name]['dim']

do_img_transform = False
saved_points = True
redo_points = False

if redo_points:
	tpoints = []
	for point in cont_points:
		t_point = XYinverse(dim_cont, a_cont, b_cont, point[0], point[1])
		tpoints.append((t_point[0]/mag, t_point[1]/mag))
		with open('tpoints.pkl', 'wb') as f:
			pickle.dump(tpoints, f)
else:
	with open('tpoints.pkl', 'rb') as f:
		tpoints = pickle.load(f)

if do_img_transform:
	img = cv2.imread(src,0)
	img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
	print(img.shape)
	if saved_points:
		with open('img_tpoints.pkl', 'rb') as f:
			results = pickle.load(f)
		with open('minmax.pkl', 'rb') as f:
			min_x, max_x, min_y, max_y = pickle.load(f)
	else:
		max_x = 0
		max_y = 0
		min_x = 0
		min_y = 0
		results = []
		for x in range(img.shape[0]):
			for y in range(img.shape[1]):
				(result_x, result_y) = XYinverse(dim_img, a_img, b_img, x, y)
				results.append((result_x, result_y))
				if max_x < result_x:
					max_x = result_x
				if max_y < result_y:
					max_y = result_y
				if min_x > result_x:
					min_x = result_x
				if min_y > result_y:
					min_y = result_y
		print('max_x: ' + str(max_x))
		print('max_y: ' + str(max_y))
		print('min_x: ' + str(min_x))
		print('min_y: ' + str(min_y))
		with open('img_tpoints.pkl', 'wb') as f:
			pickle.dump(results, f)
		with open('minmax.pkl', 'wb') as f:
			pickle.dump((min_x, max_x, min_y, max_y), f)
	t_img = np.zeros((round(max_x-min_x), round(max_y-min_y)))
	print(t_img.shape)
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			(result_x, result_y) = results[x*img.shape[1] + y]
			result_x -= min_x
			result_y -= min_y
			t_img[round(-result_x)][round(-result_y)] = img[x][y]
	t_img = cv2.rotate(t_img, cv2.ROTATE_90_CLOCKWISE)
	cv2.imwrite('transformed.jpg', t_img)
else:
	t_img = cv2.imread('transformed.jpg')
	#t_img = cv2.rotate(t_img, cv2.ROTATE_90_CLOCKWISE)
	#bgr_img = cv2.cvtColor(t_img, cv2.CV_GRAY2RBG)
	img = cv2.imread(src)
	new_img = np.zeros(img.shape)
	all_points = []
	max_x = 0
	min_x = 0
	max_y = 0
	min_y = 0
	for point in tpoints:
		x, y = (int(point[0]), int(point[1]))
		x_img = int(Xforward(dim_cont, a_img, b_img, x, y))
		y_img = int(Yforward(dim_cont, a_img, b_img, x, y))
		all_points.append((x_img, y_img))
		if max_x < x_img:
			max_x = x_img
		elif min_x > x_img:
			min_x = x_img
		elif max_y < y_img:
			max_y = y_img
		elif min_y > y_img:
			min_y = y_img
	print('max_x: ' + str(max_x))
	print('max_y: ' + str(max_y))
	print('min_x: ' + str(min_x))
	print('min_y: ' + str(min_y))
	for point in all_points:
		x, y = point
		x -= min_x
		#y -= min_y
		y = img.shape[0]-y
		new_img[y-10:y+10, x-10:x+10] = [0,0,255]
	img = img.astype(int) + new_img.astype(int)
	img[np.where(img>255)] = 255
	cv2.imwrite('img_contour.jpg', img)
	'''
	new_img = np.zeros((t_img.shape[1], t_img.shape[0], t_img.shape[2]))
	imshape = t_img.shape
	#print(tpoints)
	#TODO: Align with image (offset?)
	for point in tpoints:
		x, y = (int(point[0]), int(point[1]))
		y = imshape[0] - y
		x = imshape[1] - x
		new_img[x-30:x+30, y-30:y+30] = [0,0,255]
	new_img = cv2.rotate(new_img, cv2.ROTATE_90_CLOCKWISE)
	#ipdb.set_trace()
	t_img = t_img.astype(int) + new_img.astype(int)
	t_img[np.where(t_img>255)] = 255
	cv2.imwrite('transformed_contour.jpg', t_img)
	'''
