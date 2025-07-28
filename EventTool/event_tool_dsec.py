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


import numpy as np
import torch 

# 实现的功能包括：读取事件流->事件帧显示并存储，体素存储，图像帧与事件的结合存储


def dictionary_of_numpy_arrays_to_tensors(sample):
    """Transforms dictionary of numpy arrays to dictionary of tensors."""
    if isinstance(sample, dict):
        return {
            key: dictionary_of_numpy_arrays_to_tensors(value)
            for key, value in sample.items()
        }
    if isinstance(sample, np.ndarray):
        if len(sample.shape) == 2:
            return torch.from_numpy(sample).float().unsqueeze(0)
        else:
            return torch.from_numpy(sample).float()
    return sample

class EventSequenceToVoxelGrid_Pytorch(object):
    # Source: https://github.com/uzh-rpg/rpg_e2vid/blob/master/utils/inference_utils.py#L480
    def __init__(self, num_bins, gpu=False, gpu_nr=0, normalize=True, forkserver=True):
        if forkserver:
            try:
                torch.multiprocessing.set_start_method('forkserver')
            except RuntimeError:
                pass
        self.num_bins = num_bins
        self.normalize = normalize
        if gpu:
            if not torch.cuda.is_available():
                print('Warning: There\'s no CUDA support on this machine!')
            else:
                self.device = torch.device('cuda:' + str(gpu_nr))
        else:
            self.device = torch.device('cpu')

    def __call__(self, event_sequence):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.
        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        :param device: device to use to perform computations
        :return voxel_grid: PyTorch event tensor (on the device specified)
        """

        events = event_sequence.features.astype('float')

        width = event_sequence.image_width
        height = event_sequence.image_height

        assert (events.shape[1] == 4)
        assert (self.num_bins > 0)
        assert (width > 0)
        assert (height > 0)

        with torch.no_grad():

            events_torch = torch.from_numpy(events)
            # with DeviceTimer('Events -> Device (voxel grid)'):
            events_torch = events_torch.to(self.device)

            # with DeviceTimer('Voxel grid voting'):
            voxel_grid = torch.zeros(self.num_bins, height, width, dtype=torch.float32, device=self.device).flatten()

            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events_torch[-1, 0]
            first_stamp = events_torch[0, 0]

            assert last_stamp.dtype == torch.float64, 'Timestamps must be float64!'
            # assert last_stamp.item()%1 == 0, 'Timestamps should not have decimals'

            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events_torch[:, 0] = (self.num_bins - 1) * (events_torch[:, 0] - first_stamp) / deltaT
            ts = events_torch[:, 0]
            xs = events_torch[:, 1].long()
            ys = events_torch[:, 2].long()
            pols = events_torch[:, 3].float()
            pols[pols == 0] = -1  # polarity should be +1 / -1


            tis = torch.floor(ts)
            tis_long = tis.long()
            dts = ts - tis
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            valid_indices = tis < self.num_bins
            valid_indices &= tis >= 0

            if events_torch.is_cuda:
                datatype = torch.cuda.LongTensor
            else:
                datatype = torch.LongTensor

            voxel_grid.index_add_(dim=0,
                                  index=(xs[valid_indices] + ys[valid_indices]
                                         * width + tis_long[valid_indices] * width * height).type(
                                      datatype),
                                  source=vals_left[valid_indices])


            valid_indices = (tis + 1) < self.num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(dim=0,
                                  index=(xs[valid_indices] + ys[valid_indices] * width
                                         + (tis_long[valid_indices] + 1) * width * height).type(datatype),
                                  source=vals_right[valid_indices])

            voxel_grid = voxel_grid.view(self.num_bins, height, width)

        if self.normalize:
            mask = torch.nonzero(voxel_grid, as_tuple=True)
            if mask[0].size()[0] > 0:
                mean = voxel_grid[mask].mean()
                std = voxel_grid[mask].std()
                if std > 0:
                    voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                else:
                    voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid


class EventSlicer:
    def __init__(self, h5f: h5py.File):
        self.h5f = h5f

        self.events = dict()
        for dset_str in ['p', 'x', 'y', 't']:
            self.events[dset_str] = self.h5f['events/{}'.format(dset_str)]

        # This is the mapping from milliseconds to event index:
        # It is defined such that
        # (1) t[ms_to_idx[ms]] >= ms*1000
        # (2) t[ms_to_idx[ms] - 1] < ms*1000
        # ,where 'ms' is the time in milliseconds and 't' the event timestamps in microseconds.
        #
        # As an example, given 't' and 'ms':
        # t:    0     500    2100    5000    5000    7100    7200    7200    8100    9000
        # ms:   0       1       2       3       4       5       6       7       8       9
        #
        # we get
        #
        # ms_to_idx:
        #       0       2       2       3       3       3       5       5       8       9
        self.ms_to_idx = np.asarray(self.h5f['ms_to_idx'], dtype='int64')

        self.t_offset = int(h5f['t_offset'][()])
        self.t_final = int(self.events['t'][-1]) + self.t_offset

    def get_final_time_us(self):
        return self.t_final

    def get_events(self, t_start_us: int, t_end_us: int) -> Dict[str, np.ndarray]:
        """Get events (p, x, y, t) within the specified time window
        Parameters
        ----------
        t_start_us: start time in microseconds
        t_end_us: end time in microseconds
        Returns
        -------
        events: dictionary of (p, x, y, t) or None if the time window cannot be retrieved
        """
        assert t_start_us < t_end_us

        # We assume that the times are top-off-day, hence subtract offset:
        t_start_us -= self.t_offset
        t_end_us -= self.t_offset
        print(self.t_offset)
        t_start_ms, t_end_ms = self.get_conservative_window_ms(t_start_us, t_end_us)
        
        t_start_ms_idx = self.ms2idx(t_start_ms)
        t_end_ms_idx = self.ms2idx(t_end_ms)

        if t_start_ms_idx is None or t_end_ms_idx is None:
            # Cannot guarantee window size anymore
            return None

        events = dict()
        time_array_conservative = np.asarray(self.events['t'][t_start_ms_idx:t_end_ms_idx])
        idx_start_offset, idx_end_offset = self.get_time_indices_offsets(time_array_conservative, t_start_us, t_end_us)
        t_start_us_idx = t_start_ms_idx + idx_start_offset
        t_end_us_idx = t_start_ms_idx + idx_end_offset
        # Again add t_offset to get gps time
        events['t'] = time_array_conservative[idx_start_offset:idx_end_offset] + self.t_offset
        for dset_str in ['p', 'x', 'y']:
            events[dset_str] = np.asarray(self.events[dset_str][t_start_us_idx:t_end_us_idx])
            assert events[dset_str].size == events['t'].size
        return events


    @staticmethod
    def get_conservative_window_ms(ts_start_us: int, ts_end_us) -> Tuple[int, int]:
        """Compute a conservative time window of time with millisecond resolution.
        We have a time to index mapping for each millisecond. Hence, we need
        to compute the lower and upper millisecond to retrieve events.
        Parameters
        ----------
        ts_start_us:    start time in microseconds
        ts_end_us:      end time in microseconds
        Returns
        -------
        window_start_ms:    conservative start time in milliseconds
        window_end_ms:      conservative end time in milliseconds
        """
        assert ts_end_us > ts_start_us
        window_start_ms = math.floor(ts_start_us/1000)
        window_end_ms = math.ceil(ts_end_us/1000)
        return window_start_ms, window_end_ms
    @staticmethod
    # @jit(nopython=True)
    def get_time_indices_offsets(
            time_array: np.ndarray,
            time_start_us: int,
            time_end_us: int) -> Tuple[int, int]:
        """Compute index offset of start and end timestamps in microseconds
        Parameters
        ----------
        time_array:     timestamps (in us) of the events
        time_start_us:  start timestamp (in us)
        time_end_us:    end timestamp (in us)
        Returns
        -------
        idx_start:  Index within this array corresponding to time_start_us
        idx_end:    Index within this array corresponding to time_end_us
        such that (in non-edge cases)
        time_array[idx_start] >= time_start_us
        time_array[idx_end] >= time_end_us
        time_array[idx_start - 1] < time_start_us
        time_array[idx_end - 1] < time_end_us
        this means that
        time_start_us <= time_array[idx_start:idx_end] < time_end_us
        """

        assert time_array.ndim == 1

        idx_start = -1
        if time_array[-1] < time_start_us:
            # This can happen in extreme corner cases. E.g.
            # time_array[0] = 1016
            # time_array[-1] = 1984
            # time_start_us = 1990
            # time_end_us = 2000

            # Return same index twice: array[x:x] is empty.
            return time_array.size, time_array.size
        else:
            for idx_from_start in range(0, time_array.size, 1):
                if time_array[idx_from_start] >= time_start_us:
                    idx_start = idx_from_start
                    break
        assert idx_start >= 0

        idx_end = time_array.size
        for idx_from_end in range(time_array.size - 1, -1, -1):
            if time_array[idx_from_end] >= time_end_us:
                idx_end = idx_from_end
            else:
                break

        assert time_array[idx_start] >= time_start_us
        if idx_end < time_array.size:
            assert time_array[idx_end] >= time_end_us
        if idx_start > 0:
            assert time_array[idx_start - 1] < time_start_us
        if idx_end > 0:
            assert time_array[idx_end - 1] < time_end_us
        return idx_start, idx_end

    def ms2idx(self, time_ms: int) -> int:
        # print(time_ms)
        assert time_ms >= 0
        if time_ms >= self.ms_to_idx.size:
            return None
        return self.ms_to_idx[time_ms]




def events_to_event_image(event_sequence, height, width, background=None, rotation_angle=None, crop_window=None,
                          horizontal_flip=False, flip_before_crop=False):
    polarity = event_sequence[:, 3] == -1.0
    x_negative = event_sequence[~polarity, 1].astype(np.int)
    y_negative = event_sequence[~polarity, 2].astype(np.int)
    x_positive = event_sequence[polarity, 1].astype(np.int)
    y_positive = event_sequence[polarity, 2].astype(np.int)

    positive_histogram, _, _ = np.histogram2d(
        x_positive,
        y_positive,
        bins=(width, height),
        range=[[0, width], [0, height]])
    negative_histogram, _, _ = np.histogram2d(
        x_negative,
        y_negative,
        bins=(width, height),
        range=[[0, width], [0, height]])

    # Red -> Negative Events
    red = np.transpose((negative_histogram >= positive_histogram) & (negative_histogram != 0))
    # Blue -> Positive Events
    blue = np.transpose(positive_histogram > negative_histogram)
    # Normally, we flip first, before we apply the other data augmentations
    if flip_before_crop:
        if horizontal_flip:
            red = np.flip(red, axis=1)
            blue = np.flip(blue, axis=1)
        # Rotate, if necessary
        if rotation_angle is not None:
            red = rotate(red, angle=rotation_angle, preserve_range=True).astype(bool)
            blue = rotate(blue, angle=rotation_angle, preserve_range=True).astype(bool)
        # Crop, if necessary
        if crop_window is not None:
            tf = transformers.RandomCropping(crop_height=crop_window['crop_height'],
                                             crop_width=crop_window['crop_width'],
                                             left_right=crop_window['left_right'],
                                             shift=crop_window['shift'])
            red = tf.crop_image(red, None, window=crop_window)
            blue = tf.crop_image(blue, None, window=crop_window)
    else:
        # Rotate, if necessary
        if rotation_angle is not None:
            red = rotate(red, angle=rotation_angle, preserve_range=True).astype(bool)
            blue = rotate(blue, angle=rotation_angle, preserve_range=True).astype(bool)
        # Crop, if necessary
        if crop_window is not None:
            tf = transformers.RandomCropping(crop_height=crop_window['crop_height'],
                                             crop_width=crop_window['crop_width'],
                                             left_right=crop_window['left_right'],
                                             shift=crop_window['shift'])
            red = tf.crop_image(red, None, window=crop_window)
            blue = tf.crop_image(blue, None, window=crop_window)
        if horizontal_flip:
            red = np.flip(red, axis=1)
            blue = np.flip(blue, axis=1)

    if background is None:
        height, width = red.shape
        # background = torch.full((3, height, width), 255).byte()
        background = torch.full((3, height, width), 255, dtype=torch.long)
        
    if len(background.shape) == 2:
        background = background.unsqueeze(0)
    else:
        if min(background.size()) == 1:
            background = grayscale_to_rgb(background)
        else:
            if not isinstance(background, torch.Tensor):
                background = torch.from_numpy(background)
    points_on_background = plot_points_on_background(
        torch.nonzero(torch.from_numpy(red.astype(np.uint8))), background,
        [255, 0, 0])
    points_on_background = plot_points_on_background(
        torch.nonzero(torch.from_numpy(blue.astype(np.uint8))),
        points_on_background, [0, 0, 255])
    return points_on_background


def plot_points_on_background(points_coordinates,
                              background,
                              points_color=[0, 0, 255]):
    """
    Args:
        points_coordinates: array of (y, x) points coordinates
                            of size (number_of_points x 2).
        background: (3 x height x width)
                    gray or color image uint8.
        color: color of points [red, green, blue] uint8.
    """
    if not (len(background.size()) == 3 and background.size(0) == 3):
        raise ValueError('background should be (color x height x width).')
    _, height, width = background.size()
    background_with_points = background.clone()
    y, x = points_coordinates.transpose(0, 1)
    if len(x) > 0 and len(y) > 0: # There can be empty arrays!
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        if not (x_min >= 0 and y_min >= 0 and x_max < width and y_max < height):
            raise ValueError('points coordinates are outsize of "background" '
                             'boundaries.')
        background_with_points[:, y, x] = torch.Tensor(points_color).type_as(
            background).unsqueeze(-1)
    return background_with_points


def events_to_voxel_grid(self, evs, width, height, num_bins, use_pol=True, flip_horizontal=False):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.

        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        """
        p = events['p']
        t = events['t'].astype(np.float64)/1000000
        x = events['x']
        y = events['y']
        # extract
        # t, x, y, p = evs['t'].astype(np.float64)/1000000, evs['x'], evs['y'], evs['p']
        if not use_pol:
            p[p != 1] = 1

        if flip_horizontal:
            x = width-1 - x

        # if is_rectified:
        #     xy_rect = self.rectify_events(x, y)

            # x = np.rint(xy_rect[:, 0])
            # y = np.rint(xy_rect[:, 1])
        
        ev = np.vstack((t, x, y, p)).transpose()

        if len(ev) == 0:
                raise ValueError("Events is None!")
        
        assert(ev.shape[1] == 4)
        assert(num_bins > 0)
        assert(width > 0)
        assert(height > 0)

        # if self.is_rectified:
        #     height = self.img_height
        #     width = self.img_width
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


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--name', default='flowformer', help="name your experiment")
    # parser.add_argument('--stage', help="determines which dataset to use for training") 
    # parser.add_argument('--validation', type=str, nargs='+')

    # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # args = parser.parse_args()
    ev_data_file = "I:/DSEC/dataset_demo/events/events.h5"
    ev_location = h5py.File(str(ev_data_file), 'r')
    events_slicer = EventSlicer(ev_location)
    events = events_slicer.get_events(49599200500, 49599200500+20000)

    p = events['p'].astype(np.int8)
    t = events['t'].astype(np.float64)
    x = events['x']
    y = events['y']
    p = 2*p - 1

    events_rectified = np.stack([t, x, y, p], axis=-1)

    event_image = events_to_event_image(
            event_sequence=events_rectified,
            height=480,
            width=640
        ).numpy()
    
    # save voxel
    evs_voxel = events_to_voxel_grid(events, 3, 640, 480, 3)      
    for i in range(evs_voxel.shape[0]):
        name = '000000_' + str(i).zfill(2)
        normalized_image = cv2.normalize(evs_voxel[i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        uint8_image = (normalized_image * 255).astype(np.uint8)
        # if self.is_rectified:
        #     uint8_image = uint8_image[self.after_crop[0]:self.after_crop[1], self.after_crop[2]:self.after_crop[3]]
        #     cv2.imwrite(os.path.join(voxel_rectified_saved_path, name + '.png'), uint8_image)
        # else:
        cv2.imwrite(os.path.join("I:/DSEC/dataset_demo/event_voxel", name + '.png'), uint8_image)
                    
    # name_events = '000'
        # out_path = os.path.join()
    out_path = "I:/DSEC/dataset_demo/event_frame/000000.png"
    imageio.imsave(out_path, event_image.transpose(1,2,0))
    # 将图像和事件帧混搭覆盖
    img_raw = cv2.imread("I:/DSEC/dataset_demo/images_rectify/000000.png")
    event_image = cv2.imread("I:/DSEC/dataset_demo/event_frame/000000.png")
    # event_image = event_image.transpose(1,2,0)
    lower_white = np.array([220, 220, 220])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(event_image, lower_white, upper_white)
    event_image[mask != 0] = [0, 0, 0]
    # print(event_image.shape)
    # print(img_raw.shape)
    overlay_image = cv2.add(img_raw, event_image)
    cv2.imwrite(os.path.join("I:/DSEC/dataset_demo", '000000.png'), overlay_image)


