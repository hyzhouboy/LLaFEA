from __future__ import print_function, division
from typing import Dict, Tuple

from pathlib import Path
from skimage.transform import rotate, warp

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import hdf5plugin
import argparse
from tqdm import tqdm
import numpy as np
# import torch 
import zipfile

from h5_tools.event_packagers import *
from metavision_core.event_io.raw_reader import RawReader
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm

from PIL import Image

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


class EventCamera:
    def __init__(self, param):
        def getRectifyTransform(height, width, alpha=1):
            def getForwardMap(K, D, R, P, W, H):
                tempmap = np.zeros((W, H))
                WHs_sep = np.where(tempmap == 0)
                points_raw = np.vstack((WHs_sep[0], WHs_sep[1])).astype(np.float64)
                points_new = np.squeeze(cv2.undistortPoints(points_raw, K, D, R=R, P=P))
                maps_f = np.reshape(points_new, (W, H, 2))
                return maps_f
            left_K = self.cam_matrix_left
            right_K = self.cam_matrix_right
            left_distortion = self.distortion_l
            right_distortion = self.distortion_r
            R = self.R
            T = self.T
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, (width, height), R, T, alpha=alpha)
            map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
            map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)
            maps1_f = getForwardMap(left_K, left_distortion, R1, P1, width, height)
            maps2_f = getForwardMap(right_K, right_distortion, R2, P2, width, height)
            return map1x, map1y, map2x, map2y, maps1_f, maps2_f 
        
        self.events_color = param.events_color
        # self.data_formate = param.data_formate
        self.n_bins = param.n_bins
        self.use_polarity = param.use_polarity
        self.slice_by_time = param.slice_by_time
        self.delta_t = param.delta_t
        self.n_events = param.n_events
        self.is_saving_h5 = param.is_saving_h5
        self.is_flip = param.is_flip
        self.timestamp_file = param.timestamp_file
        self.events_delta = param.events_delta
        self.is_rectified = param.is_rectified

        self.img_height = param.img_height
        self.img_width = param.img_width
        self.offset_x = param.offset_x
        self.offset_y = param.offset_y
        self.after_crop = param.after_crop


        # 相机内外参
        # 帧相机内参
        self.cam_matrix_left = param.cam_matrix_left
        # 事件相机内参
        self.cam_matrix_right = param.cam_matrix_right
 
        # 帧和事件相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = param.distortion_l
        self.distortion_r = param.distortion_r
 
        # 旋转矩阵（事件到帧）
        self.R = param.R
        # self.R = np.linalg.inv(self.R)
 
        # 平移矩阵（事件到帧）
        self.T = param.T
        # self.T = -1 * self.T

        # 主点列坐标的差
        self.doffs = param.doffs
        # self.mv_iterator = []

        self.map1x, self.map1y, self.map2x, self.map2y, self.maps1_f, self.maps2_f = getRectifyTransform(self.img_height, self.img_width)

    def read_data(self, ev_data_file, im_data_path):
        def get_timestamp(file_name):
            timestamp = []
            with open(file_name, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        start_t = int(line.strip().split()[0])/1000
                        continue
                    t = int(line.strip().split()[0])/1000 - start_t
                    timestamp.append(str(int(t)) + '\n')
            return timestamp
        self.event_data_file = ev_data_file
        self.record_raw = RawReader(self.event_data_file)
        self.height, self.width = self.record_raw.get_size()
        mv_iterator = []
        if self.timestamp_file == None:
            if self.slice_by_time:
                mv_iterator = EventsIterator(input_path=self.event_data_file, mode='delta_t', delta_t=self.delta_t)
            else:
                mv_iterator = EventsIterator(input_path=self.event_data_file, mode='n_events', delta_t=self.n_events)
        else:
            timestamp_list = get_timestamp(self.timestamp_file)
            for idx in range(len(timestamp_list)):
                print('Now is read: ' + str(idx))
                if idx < len(timestamp_list)-1:
                    timestamp = int(timestamp_list[idx].strip('\n'))
                    # image
                    image_name = im_data_path + '/image_' + str(idx+2).zfill(8) + '.bmp'
                    mv_iterator.append([timestamp, image_name])
        
        return mv_iterator
    

    def rectify_events(self, x: np.ndarray, y: np.ndarray):
        return self.maps2_f[x, y]
        
    def events_to_voxel_grid(self, events, width, height, num_bins, use_pol=True, flip_horizontal=False):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.

        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        """
        
        # extract
        t, x, y, p = events['t']/1000000, events['x'], events['y'], events['p']
        if not use_pol:
            p[p != 1] = 1

        if flip_horizontal:
            x = width-1 - x

        if self.is_rectified:
            xy_rect = self.rectify_events(x, y)

            x = np.rint(xy_rect[:, 0])
            y = np.rint(xy_rect[:, 1])
        
        ev = np.vstack((t, x, y, p)).transpose()

        if len(ev) == 0:
                raise ValueError("Events is None!")
        
        assert(ev.shape[1] == 4)
        assert(num_bins > 0)
        assert(width > 0)
        assert(height > 0)

        if self.is_rectified:
            height = self.img_height
            width = self.img_width
        voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()
        
        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = ev[-1, 0]
        first_stamp = ev[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        ev[:, 0] = (num_bins - 1) * (ev[:, 0] - first_stamp) / deltaT
        ts = ev[:, 0]
        xs = ev[:, 1].astype(np.int)
        ys = ev[:, 2].astype(np.int)
        pols = ev[:, 3]
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

    def events_to_frames(self, events, width, height, ev_color, flip_horizontal=False):
        counter = events.size
        image = None
        if counter == 0:
            raise ValueError("Events is None!")
        
        t, x, y, p = events['t'], events['x'], events['y'], events['p']
        if flip_horizontal:
            x = width - 1 - x

        if self.is_rectified:
            xy_rect = self.rectify_events(x, y)

            x = np.rint(xy_rect[:, 0]).astype(np.int)
            y = np.rint(xy_rect[:, 1]).astype(np.int)

        if ev_color == 'Gray':
            img = np.ones(shape=(height, width), dtype=int) * 255
            if self.is_rectified:
                img = np.ones(shape=(self.img_height, self.img_width), dtype=int) * 255
            for j in range(events.size):
                img[y[j], x[j]] = 0
            image = img
        elif ev_color == 'RB':
            img = np.ones(shape=(height, width), dtype=int) * 0.5
            if self.is_rectified:
                img = np.ones(shape=(self.img_height, self.img_width), dtype=int) * 0.5
            for j in range(events.size):
                img[y[j], x[j]] += (2 * p[j] - 1) * 0.25  # p is [0, 1], convert it to [-0.25, 0.25], only keep last p; img: [0.5, 0.75(positive), 0.25(negative)]
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
        elif ev_color == 'WB':
            image = np.zeros((height, width, 3), dtype=np.uint8)
            BaseFrameGenerationAlgorithm.generate_frame(events, image)
        
        return image


    def write_event(self, mv_iterator):
        output_path = os.path.join(os.path.dirname(self.event_data_file), os.path.basename(self.event_data_file).replace('.raw', ''))
        frame_save_path = output_path + "_frame"
        voxel_save_path = output_path + "_voxel"
        h5_save_path = output_path + ".h5"
        if not os.path.exists(frame_save_path):
            os.makedirs(frame_save_path)

        if not os.path.exists(voxel_save_path):
            os.makedirs(voxel_save_path)
        
        if self.is_saving_h5:
            ep = hdf5_packager(h5_save_path)
            first_ts = -1
            t0 = -1
            sensor_size = [self.height, self.width]
            # Extract events to h5
            ep.set_data_available(num_images=0, num_flow=0)
            total_num_pos, total_num_neg, last_ts = 0, 0, 0
            slice_num = 0
            event_index = 0
            event_indices_list = []
        
        for frame_id, evs in enumerate(tqdm(mv_iterator)):
            print(frame_id)
            image_name = str(frame_id).zfill(5)
            
            if self.is_saving_h5:
                evs['p'][evs['p'] < 0] = 0
                
                evs['p'] = evs['p'].astype(bool)
                if first_ts == -1:
                    first_ts = evs['t'][0]
                last_ts = evs['t'][-1]
                sum_ps = sum(evs['p'])
                total_num_pos += sum_ps
                total_num_neg += len(evs['p']) - sum_ps
                ep.package_events(evs)
                tmp_index = event_index
                event_index += len(evs['t'])
                event_indices_list.append([tmp_index, event_index])
                slice_num += 1

            evs_image = self.events_to_frames(evs, self.width, self.height, self.events_color, self.is_flip)
            cv2.imwrite(os.path.join(frame_save_path, image_name + '.png'), evs_image)
            
            evs_voxel = self.events_to_voxel_grid(evs, self.width, self.height, self.n_bins, self.use_polarity, self.is_flip)
            for i in range(evs_voxel.shape[0]):
                name = image_name + '_' + str(i).zfill(2)
                
                normalized_image = cv2.normalize(evs_voxel[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                uint8_image = (normalized_image * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(voxel_save_path, name + '.png'), uint8_image)
               
        if self.is_saving_h5:
            t0 = first_ts
            ep.add_metadata(total_num_pos, total_num_neg, last_ts - t0, t0, last_ts, num_imgs=0, num_flow=0, sensor_size=sensor_size)
        

    def write_evecam(self, mv_iterator):
        output_path = os.path.join(os.path.dirname(self.event_data_file), os.path.basename(self.event_data_file).replace('.raw', ''))
        frame_save_path = output_path + "_frame"
        voxel_save_path = output_path + "_voxel"
        h5_save_path = output_path + ".h5"
        frame_rectified_saved_path = output_path + "_frame_rectified"
        voxel_rectified_saved_path = output_path + "_voxel_rectified"
        image_rectified_saved_path = os.path.join(os.path.dirname(self.event_data_file), 'left_image_rectified')
        evecam_overlay_saved_path = os.path.join(os.path.dirname(self.event_data_file), 'image_event_overlay')
        if self.is_rectified:
            if not os.path.exists(frame_rectified_saved_path):
                os.makedirs(frame_rectified_saved_path)
            
            if not os.path.exists(voxel_rectified_saved_path):
                os.makedirs(voxel_rectified_saved_path)
            if not os.path.exists(image_rectified_saved_path):
                os.makedirs(image_rectified_saved_path)
            if not os.path.exists(evecam_overlay_saved_path):
                os.makedirs(evecam_overlay_saved_path)
        

        if not os.path.exists(frame_save_path):
            os.makedirs(frame_save_path)

        if not os.path.exists(voxel_save_path):
            os.makedirs(voxel_save_path)
        
        if self.is_saving_h5:
            ep = hdf5_packager(h5_save_path)
            first_ts = -1
            t0 = -1
            sensor_size = [self.height, self.width]
            # Extract events to h5
            ep.set_data_available(num_images=0, num_flow=0)
            total_num_pos, total_num_neg, last_ts = 0, 0, 0
            slice_num = 0
            event_index = 0
            event_indices_list = []
        
        for frame_id, [evs_time, img_name] in enumerate(tqdm(mv_iterator)):
            if self.is_rectified:
                img_raw = cv2.imread(img_name)
                img_crop = img_raw[self.offset_y:self.offset_y+self.img_height, self.offset_x:self.offset_x+self.img_width]
                img_rectified = cv2.remap(img_crop, self.map1x, self.map1y, cv2.INTER_AREA)
                img_rectified_crop = img_rectified[self.after_crop[0]:self.after_crop[1], self.after_crop[2]:self.after_crop[3]]
                cv2.imwrite(os.path.join(image_rectified_saved_path, os.path.basename(img_name).replace('.bmp', '.png')), img_rectified_crop)

            # read event data
            self.record_raw.seek_time(evs_time - self.events_delta)
            evs = self.record_raw.load_delta_t(self.events_delta)
            print(frame_id)
            image_name = str(frame_id+2).zfill(8)
            
            if self.is_saving_h5:
                evs['p'][evs['p'] < 0] = 0
                
                evs['p'] = evs['p'].astype(bool)
                if first_ts == -1:
                    first_ts = evs['t'][0]
                last_ts = evs['t'][-1]
                sum_ps = sum(evs['p'])
                total_num_pos += sum_ps
                total_num_neg += len(evs['p']) - sum_ps
                ep.package_events(evs)
                tmp_index = event_index
                event_index += len(evs['t'])
                event_indices_list.append([tmp_index, event_index])
                slice_num += 1

            # save ev_img
            evs_image = self.events_to_frames(evs, self.width, self.height, self.events_color, self.is_flip)
            if self.is_rectified:
                evs_image = evs_image[self.after_crop[0]:self.after_crop[1], self.after_crop[2]:self.after_crop[3]]
                cv2.imwrite(os.path.join(frame_rectified_saved_path, image_name + '.png'), evs_image)
                # 将图像和事件帧混搭覆盖
                evs_image[np.all(evs_image == (255, 255, 255), axis=-1)] = (0, 0, 0)
                overlay_image = cv2.add(img_rectified_crop, evs_image)
                cv2.imwrite(os.path.join(evecam_overlay_saved_path, image_name + '.png'), overlay_image)
            else:
                cv2.imwrite(os.path.join(frame_save_path, image_name + '.png'), evs_image)

            # save voxel
            evs_voxel = self.events_to_voxel_grid(evs, self.width, self.height, self.n_bins, self.use_polarity, self.is_flip)      
            for i in range(evs_voxel.shape[0]):
                name = image_name + '_' + str(i).zfill(2)
                normalized_image = cv2.normalize(evs_voxel[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                uint8_image = (normalized_image * 255).astype(np.uint8)
                if self.is_rectified:
                    uint8_image = uint8_image[self.after_crop[0]:self.after_crop[1], self.after_crop[2]:self.after_crop[3]]
                    cv2.imwrite(os.path.join(voxel_rectified_saved_path, name + '.png'), uint8_image)
                else:
                    cv2.imwrite(os.path.join(voxel_save_path, name + '.png'), uint8_image)
               
        if self.is_saving_h5:
            t0 = first_ts
            ep.add_metadata(total_num_pos, total_num_neg, last_ts - t0, t0, last_ts, num_imgs=0, num_flow=0, sensor_size=sensor_size)
        

    def write(self, mv_iterator):
        if self.timestamp_file == None:
            self.write_event(mv_iterator)
        else:
            self.write_evecam(mv_iterator)
    
            










