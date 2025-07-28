#批量处理hdf5转mat
import h5py
import numpy as np
import os
import scipy.io as io
from pathlib import Path
inputpath='E:/pythonProject/AEDNet/data/AEDNetDataset_BA/test/'
outputpath='E:/pythonProject/AEDNet/data/AEDNetDataset_BA/test/'
#dirname, filename = os.path.split('D:/博士课题/事件相机数据集/DVSCLEAN/real_data/indoor2/indoor6.hdf5')
doc_name = os.listdir(inputpath)
#print(str(doc_name[0]))
#filename=str(path)+str(doc_name[0])
for i in np.arange(len(doc_name)):
 filename = str(inputpath) + str(doc_name[i])
 name = doc_name[i].split('.')[0]
 print(name)
 filename2 = str(outputpath) + str(name)
 a=h5py.File(filename,'r')['events/polarity']
 b=h5py.File(filename,'r')['events/timestamp']
 c=h5py.File(filename,'r')['events/x']
 d=h5py.File(filename, 'r')['events/y']
 e=h5py.File(filename, 'r')['events/label']
 #d=h5py.File('D:/博士课题/事件相机数据集/DVSCLEAN/real_data/indoor2/indoor6.hdf5','r')['events/y']
 a=np.array([a]).T
 b=np.array([b]).T
 c=np.array([c]).T
 d=np.array([d]).T
 e=np.array([e]).T
 events = np.hstack((e, c, d, b, a))
 print(events[0:5])
 #events=np.hstack((c,d,b,a))
#clip=events[0:1000000,:]
 clip=events[:,:]
 #print(clip.shape)
 #np.save(filename2, clip) %保存成npy文件
 io.savemat('E:/pythonProject/AEDNet/data/AEDNetTestset_BA/MAH00444_100.mat', {'tuple': clip})
#print(events.shape)
#test = np.load('E:/pythonProject/AEDNet/data/AEDNetDataset_BA/events.npy')
#np.savetxt('E:/pythonProject/AEDNet/data/AEDNetDataset_BA/events2.txt', test, fmt='%s', newline='\n')
