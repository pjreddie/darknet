import os
import re

base_dir = '/home/icu_data/data/video'
txt_dir = '/home/ejchou/darknet/frame_dirs.txt'

with open(txt_dir, 'w') as f:
	for instance in os.listdir(base_dir):
		instance_pth = os.path.join(base_dir, instance)
		if os.path.isdir(instance_pth):
			# print(os.listdir(instance_pth)[0])
			# print(len(os.listdir(instance_pth)))
			for frame in os.listdir(instance_pth):
				frame_pth = os.path.join(instance_pth, frame)
				if re.search('.jpg', frame_pth) or re.search('.png', frame_pth): 
					f.write(frame_pth + '\n')
