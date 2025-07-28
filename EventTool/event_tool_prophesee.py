from __future__ import print_function, division
import sys
# sys.path.append('core')
import math
from typing import Dict, Tuple
from pathlib import Path
import weakref
from skimage.transform import rotate, warp
import h5py
import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
import hdf5plugin
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

from tqdm import tqdm
import numpy as np
# import torch 
import zipfile

from h5_tools.event_packagers import *
from metavision_core.event_io.raw_reader import RawReader
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm
# 实现的功能包括：读取事件流->事件帧显示并存储，体素存储，图像帧与事件的结合存储


# 事件流转体素
def events_to_voxel_grid(events, num_bins, width, height):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.

    :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
    :param num_bins: number of bins in the temporal axis of the voxel grid
    :param width, height: dimensions of the voxel grid
    """

    assert(events.shape[1] == 4)
    assert(num_bins > 0)
    assert(width > 0)
    assert(height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT
    ts = events[:, 0]
    xs = events[:, 1].astype(np.int)
    ys = events[:, 2].astype(np.int)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + tis[valid_indices] * width * height, vals_left[valid_indices])

    valid_indices = (tis + 1) < num_bins
    np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width
              + (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))

    return voxel_grid


class FixedSizeEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each containing a fixed number of events.
    """

    def __init__(self, path_to_event_file, num_events=10000, start_index=0):
        print('Will use fixed size event windows with {} events'.format(num_events))
        print('Output frame rate: variable')
        self.iterator = pd.read_csv(path_to_event_file, delim_whitespace=True, header=None,
                                    names=['t', 'x', 'y', 'pol'],
                                    dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'pol': np.int16},
                                    engine='c',
                                    skiprows=start_index + 1, chunksize=num_events, nrows=None, memory_map=True)

    def __iter__(self):
        return self

    def __next__(self):
        event_window = self.iterator.__next__().values
        return event_window


class FixedDurationEventReader:
    """
    Reads events from a '.txt' or '.zip' file, and packages the events into
    non-overlapping event windows, each of a fixed duration.

    **Note**: This reader is much slower than the FixedSizeEventReader.
              The reason is that the latter can use Pandas' very efficient chunk-based reading scheme implemented in C.
    """

    def __init__(self, path_to_event_file, duration_ms=50.0, start_index=0):
        print('Will use fixed duration event windows of size {:.2f} ms'.format(duration_ms))
        print('Output frame rate: {:.1f} Hz'.format(1000.0 / duration_ms))
        file_extension = splitext(path_to_event_file)[1]
        assert(file_extension in ['.txt', '.zip'])
        self.is_zip_file = (file_extension == '.zip')

        if self.is_zip_file:  # '.zip'
            self.zip_file = zipfile.ZipFile(path_to_event_file)
            files_in_archive = self.zip_file.namelist()
            assert(len(files_in_archive) == 1)  # make sure there is only one text file in the archive
            self.event_file = self.zip_file.open(files_in_archive[0], 'r')
        else:
            self.event_file = open(path_to_event_file, 'r')

        # ignore header + the first start_index lines
        for i in range(1 + start_index):
            self.event_file.readline()

        self.last_stamp = None
        self.duration_s = duration_ms / 1000.0

    def __iter__(self):
        return self

    def __del__(self):
        if self.is_zip_file:
            self.zip_file.close()

        self.event_file.close()

    def __next__(self):
        event_list = []
        for line in self.event_file:
            if self.is_zip_file:
                line = line.decode("utf-8")
            t, x, y, pol = line.split(' ')
            t, x, y, pol = float(t), int(x), int(y), int(pol)
            event_list.append([t, x, y, pol])
            if self.last_stamp is None:
                self.last_stamp = t
            if t > self.last_stamp + self.duration_s:
                self.last_stamp = t
                event_window = np.array(event_list)
                return event_window

        raise StopIteration


# 存储事件帧图像
def draw_events_prophesee(save_path, evs_num, idx, evs_w, evs_h, evs_x, evs_y, evs_p, color, events):
    if color == 'GRAY':
        img = np.ones(shape=(evs_h, evs_w), dtype=int) * 255
        for j in range(evs_num):
            # img[y1[j], x1[j]] = (2*p1[j]-1)  # p is [0, 1], convert it to [-1,1], only keep last p
            img[evs_y[j], evs_x[j]] = 0
        image = img
    elif color == 'RB':
        img = np.ones(shape=(evs_h, evs_w), dtype=int) * 0.5
        for j in range(evs_num):
            img[evs_y[j], evs_x[j]] += (2 * evs_p[j] - 1) * 0.25  # p is [0, 1], convert it to [-0.25, 0.25], only keep last p; img: [0.5, 0.75(positive), 0.25(negative)]
        # convert img to red & blue map
        tmp0 = (img * 255).astype(np.uint8)
        tmp1 = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        rgbArray = np.zeros((tmp1.shape[0], tmp1.shape[1], tmp1.shape[2]), 'uint8')
        tmp1[tmp1 == 127] = 0
        B = tmp0.copy()
        G = tmp0.copy()
        R = tmp0.copy()

        B[tmp0 > 127] = 0
        B[tmp0 <= 127] = 255
        rgbArray[:, :, 0] = B

        G[tmp0 != 127] = 0
        G[tmp0 == 127] = 255
        rgbArray[:, :, 1] = G

        R[tmp0 >= 127] = 255
        R[tmp0 < 127] = 0
        rgbArray[:, :, 2] = R
        image = rgbArray.astype(np.uint8)
    elif color == 'WB':
        image = np.zeros((evs_h, evs_w, 3), dtype=np.uint8)
        BaseFrameGenerationAlgorithm.generate_frame(events, image)

    name = str(idx).zfill(5)
    cv2.imwrite(os.path.join(save_path, name + '.png'), image)



def event_frame(raw_path, data_iterator, h, w, file, save_im, save_np, save_h5file, flip_horizontal, ev_color):
    frame_path = os.path.join(os.path.dirname(raw_path), os.path.basename(raw_path).replace('.raw', '') + '_' + file)
    frame_save_path = os.path.join(frame_path, 'event_frame')
    np_path = os.path.join(frame_path, 'npy')
    h5_path = os.path.join(frame_path, 'h5')
    if save_im:
        if not os.path.exists(frame_save_path):
            os.makedirs(frame_save_path)
    if save_np:
        if not os.path.exists(np_path):
            os.makedirs(np_path)
    if save_h5file:
        if not os.path.exists(h5_path):
            os.makedirs(h5_path)
        h5_file = os.path.join(h5_path, os.path.basename(raw_path).replace('.raw', '.h5'))
        ep = hdf5_packager(h5_file)
        first_ts = -1
        t0 = -1
        sensor_size = [h, w]
        # Extract events to h5
        ep.set_data_available(num_images=0, num_flow=0)
        total_num_pos, total_num_neg, last_ts = 0, 0, 0
        slice_num = 0
        event_index = 0
        event_indices_list = []

    for frame_id, evs in enumerate(tqdm(data_iterator)):
        # if frame_id < 170 or frame_id > 179:
        #     continue
        # evs: x, y, p[0, 1], t[us]
        counter = evs.size
        if counter == 0:
            continue
        t, x, y, p = evs['t'], evs['x'], evs['y'], evs['p']

        # 删除越界的事件(传感器记录错误)
        delete_id = np.vstack((np.argwhere(x > w-1), np.argwhere(x < 0), np.argwhere(y > h-1), np.argwhere(y < 0)))
        t = np.delete(t, delete_id)
        x = np.delete(x, delete_id)
        y = np.delete(y, delete_id)
        p = np.delete(p, delete_id)
        counter = len(t)
        name = str(frame_id).zfill(5)
        if flip_horizontal:
            x = 1279 - x
        if save_np:
            np.save(os.path.join(np_path, name + '.npy'), np.array([x, y, t, p], dtype=np.int64))
        if save_h5file:
            p[p < 0] = 0  # should be [0 or 1]
            p = p.astype(bool)
            if first_ts == -1:
                first_ts = t[0]
            last_ts = t[-1]
            sum_ps = sum(p)
            total_num_pos += sum_ps
            total_num_neg += len(p) - sum_ps
            ep.package_events(x, y, t, p)
            tmp_index = event_index
            event_index += counter
            event_indices_list.append([tmp_index, event_index])
            slice_num += 1
        if save_im:
            draw_events_prophesee(frame_save_path, counter, frame_id, w, h, x, y, p, ev_color, evs)
    if save_h5file:
        t0 = first_ts
        ep.add_metadata(total_num_pos, total_num_neg, last_ts - t0, t0, last_ts, num_imgs=0, num_flow=0, sensor_size=sensor_size)
        # ep.add_indices(event_indices_list)



def event_voxel(raw_path, data_iterator, num_bins, h, w, file, save_im, use_pol):
    frame_path = os.path.join(os.path.dirname(raw_path), os.path.basename(raw_path).replace('.raw', '') + '_' + file)
    voxel_path = os.path.join(frame_path, 'voxel')
    if save_im:
        if not os.path.exists(voxel_path):
            os.makedirs(voxel_path)
    for frame_id, evs in enumerate(tqdm(data_iterator)):
        t, x, y, p = evs['t']/1000000, evs['x'], evs['y'], evs['p']
        # 生成体素是否考虑极性
        if not use_pol:
            p[p != 1] = 1
        events = np.vstack((t, x, y, p)).transpose()
        if len(events) == 0:
            continue
        event_tensor = events_to_voxel_grid(events,
                                            num_bins=num_bins,
                                            width=w,
                                            height=h)
        for i in range(event_tensor.shape[0]):
            name = str(frame_id).zfill(5) + '_' + str(i).zfill(2)
            if save_im:
                plt.figure(os.path.join(voxel_path, name + '.png'))
                plt.imshow(event_tensor[i])
                # plt.colorbar()  # 添加右侧取值样条，可注释
                plt.savefig(os.path.join(voxel_path, name + '.png'), dpi=300)
                plt.close()
                # cv2.imwrite(os.path.join(voxel_path, name + '.png'), event_tensor[i])



def read_file_prophesee(path, formate, bins, slice_time, use_p, dt, n, s_img, s_npy, s_h5, flip_im, event_color):
    # open a file
    record_raw = RawReader(path)
    # print(record_raw)
    height, width = record_raw.get_size()
    # number of events to generate a frame
    if slice_time:
        file_name = str(dt/1000) + 'ms'
        mv_iterator = EventsIterator(input_path=path, mode='delta_t', delta_t=dt)
    else:
        file_name = str(n)
        mv_iterator = EventsIterator(input_path=path, mode='n_events', n_events=n)
    if 'frame' in formate:
        event_frame(path,  mv_iterator, height, width, file_name, s_img, s_npy, s_h5, flip_im, event_color)
    if 'voxel' in formate:
        event_voxel(path,  mv_iterator, bins, height, width, file_name, s_img, use_p)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--name', default='flowformer', help="name your experiment")
    # parser.add_argument('--stage', help="determines which dataset to use for training") 
    # parser.add_argument('--validation', type=str, nargs='+')

    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # args = parser.parse_args()

    events_color = 'RB'  # events color(red(+) and blue(-))
    # events_color = 'WB'
    data_formate = ['frame']  # 保存事件数据的类型(事件帧或体素)
    # data_formate = ['voxel']
    n_bins = 5  # 生成体素的个数
    use_polarity = True  # 生成体素时是否考虑极性，如不考虑则全部按1处理
    slice_by_time = True  # 使用固定时间分割，False为使用固定数量间隔
    delta_t = 15000  # 当按固定时间分割时，每个片段的间隔时间(us)
    n_events = 1000000  # 当按固定数量分割时，每个片段的事件数量
    save_img = True
    save_npy = False
    save_h5 = False
    flip_events = True  # 水平翻转事件
    
    ev_data_file = "F:/Dateset/NIPS_Data/event_left.raw"
    # ev_location = h5py.File(str(ev_data_file), 'r')
    read_file_prophesee(ev_data_file, data_formate, n_bins, slice_by_time, use_polarity, delta_t, n_events, save_img, save_npy, save_h5, flip_events, events_color)
    
    
    
    
    
    # events_slicer = EventSlicer(ev_location)
    # events = events_slicer.get_events(53192302719, 53192302719+5000)

    # p = events['p'].astype(np.int8)
    # t = events['t'].astype(np.float64)
    # x = events['x']
    # y = events['y']
    # p = 2*p - 1

    # events_rectified = np.stack([t, x, y, p], axis=-1)

    # event_image = events_to_event_image(
    #         event_sequence=events_rectified,
    #         height=360,
    #         width=640
    #     ).numpy()
    
    # # name_events = '000'
    #     # out_path = os.path.join()
    # out_path = "output/event_000.png"
    # imageio.imsave(out_path, event_image.transpose(1,2,0))


