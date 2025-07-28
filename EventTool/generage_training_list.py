from __future__ import print_function, division
import sys
# sys.path.append('core')
import math
import os

name = 'training.txt'
root_dir = 'I:/DSEC'
PATH = os.listdir(root_dir)
# 遍历根目录下的所有文件夹
with open(os.path.join(root_dir, name), "w") as f:
    for dir_name in PATH:
        # 组装当前文件夹的完整路径
        current_dir_path = os.path.join(root_dir, dir_name)
        image_path = os.path.join(dir_name, 'images_rectify')
        event_path = os.path.join(dir_name, 'event_voxel')
        flow_path = os.path.join(dir_name, 'flow')
        for flow_name in os.listdir(os.path.join(current_dir_path, 'flow')):
            frame_id = int(flow_name.split('.')[0])
            image1_name = os.path.join(image_path, str(frame_id).zfill(6))
            if not os.path.exists(os.path.join(root_dir, image1_name+'.png')):
                continue
            image2_name = os.path.join(image_path, str(frame_id+2).zfill(6))
            if not os.path.exists(os.path.join(root_dir, image2_name+'.png')):
                continue
            event1_name = os.path.join(event_path, str(frame_id).zfill(6))
            event2_name = os.path.join(event_path, str(frame_id+2).zfill(6))
            if not os.path.exists(os.path.join(root_dir, event2_name+'_01.png')):
                continue
            flow_name = os.path.join(flow_path, str(frame_id).zfill(6))
        
            line = image1_name + ' ' + image2_name + ' ' + event1_name + ' ' + event2_name + ' ' + flow_name + '\n'
            line = line.replace('\\', '/')
            f.write(line)
    
    f.close()
