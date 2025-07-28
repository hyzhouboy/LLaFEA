import argparse
import pickle

import h5py
import numpy as np
import os, shutil
import json


def get_npy(data_path):
    '''
    find imgs files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['npy']
    isroot = True
    for parent, dirnames, filenames in os.walk(data_path):
        if isroot:
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
        isroot = False
    print(('Find {} .npy files'.format(len(files))))
    files.sort()
    return files


def find_safe_alternative(output_base_path):
    i = 0
    alternative_path = "{}_{:09d}".format(output_base_path, i)
    while(os.path.exists(alternative_path)):
        i += 1
        alternative_path = "{}_{:09d}".format(output_base_path, i)
        assert(i < 999999999)
    return alternative_path

def save_additional_data_as_mmap(f, mmap_pth, data):
    data_path = os.path.join(mmap_pth, data['mmap_filename'])
    data_ts_path = os.path.join(mmap_pth, data['mmap_ts_filename'])
    data_event_idx_path = os.path.join(mmap_pth, data['mmap_event_idx_filename'])
    data_key = data['h5_key']
    print('Writing {} to mmap {}, timestamps to {}'.format(data_key, data_path, data_ts_path))
    h, w, c = 1, 1, 1
    if data_key in f.keys():
        num_data = len(f[data_key].keys())
        if num_data > 0:
            data_keys = list(f[data_key].keys())
            data_size = f[data_key][data_keys[0]].attrs['size']
            h, w = data_size[0], data_size[1]
            c = 1 if len(data_size) <= 2 else data_size[2]
    else:
        num_data = 1

    # mmp_imgs = np.memmap(data_path, dtype='uint8', mode='w+', shape=(num_data, h, w, c))
    # mmp_img_ts = np.memmap(data_ts_path, dtype='float64', mode='w+', shape=(num_data, 1))
    # mmp_event_indices = np.memmap(data_event_idx_path, dtype='uint16', mode='w+', shape=(num_data, 1))

    if data_key in f.keys():
        data = []
        data_timestamps = []
        data_event_index = []
        for img_key in f[data_key].keys():
            data.append(f[data_key][img_key][:])  # image: (h, w, 1)
            data_timestamps.append(f[data_key][img_key].attrs['timestamp'])
            # data_event_index.append(f[data_key][img_key].attrs['event_idx'])

        data_stack = np.expand_dims(np.stack(data), axis=3).astype('uint8')  # (n, h, w, c)
        # data_ts_stack = np.expand_dims(np.stack(data_timestamps), axis=1).astype('float64')
        data_ts_stack = np.stack(data_timestamps).astype('float64')

        # data_event_indices_stack = np.expand_dims(np.stack(data_event_index), axis=1)
        # mmp_imgs[...] = data_stack
        # mmp_img_ts[...] = data_ts_stack
        # mmp_event_indices[...] = data_event_indices_stack

        np.save(data_path, data_stack)
        np.save(data_ts_path, data_ts_stack)


def save_flow_as_mmap(f, mmap_pth, input_flow_dir, data):
    data_path = os.path.join(mmap_pth, data['mmap_filename'])
    data_key = data['h5_key']
    print('Writing {} to mmap {}'.format(data_key, data_path))

    npy_files = get_npy(input_flow_dir)
    num_data = len(npy_files)
    (c, h, w) = np.load(npy_files[0]).shape

    data = [np.zeros((c, h, w))]
    # mmp_flows = np.memmap(data_path, dtype='uint8', mode='w+', shape=(num_data+1, c, h, w))
    for npy in npy_files:
        data.append(np.load(npy).astype(np.uint8))

    # data_stack = np.stack(data)
    # mmp_flows[...] = data_stack

    np.save(data_path, np.stack(data))


def write_metadata(f, metadata_path):
    metadata = {}
    for attr in f.attrs:
        val = f.attrs[attr]
        if isinstance(val, np.ndarray):
            val = val.tolist()
        if type(val) == np.uint32 or type(val) == np.int64:
            val = int(val)
        metadata[attr] = val
    with open(metadata_path, 'w') as js:
        json.dump(metadata, js)


def h5_to_memmap(h5_file_path, input_flow_path, output_pth):
    # mmap_pth = os.path.join(output_pth, "memmap")
    mmap_pth = output_pth
    if not os.path.exists(mmap_pth):
        os.makedirs(mmap_pth)

    ts_path = os.path.join(mmap_pth, 't.npy')
    xy_path = os.path.join(mmap_pth, 'xy.npy')
    ps_path = os.path.join(mmap_pth, 'p.npy')
    metadata_path = os.path.join(mmap_pth, 'metadata.json')

    additional_data = {
            "images":
                {
                    'h5_key' : 'images',
                    'mmap_filename' : 'images.npy',
                    'mmap_ts_filename' : 'timestamps.npy',
                    'mmap_event_idx_filename' : 'image_event_indices.npy',
                    'dims' : 3
                },
            "flow":
                {
                    'h5_key' : 'flow',
                    'mmap_filename': 'flow.npy',
                    'mmap_ts_filename': 'flow_timestamps.npy',
                    'mmap_event_idx_filename': 'flow_event_indices.npy',
                    'dims' : 3
                }
    }

    with h5py.File(h5_file_path, 'r') as f:
        num_events = f.attrs['num_events']
        num_images = f.attrs['num_imgs']
        num_flow = f.attrs['num_flow']

        # mmp_ts = np.memmap(ts_path, dtype='float64', mode='w+', shape=(num_events, 1))
        # mmp_xy = np.memmap(xy_path, dtype='int16', mode='w+', shape=(num_events, 2))
        # mmp_ps = np.memmap(ps_path, dtype='uint8', mode='w+', shape=(num_events, 1))

        # mmp_ts[:, 0] = f['events/ts'][:]
        # mmp_xy[:, :] = np.stack((f['events/xs'][:], f['events/ys'][:])).transpose()
        # mmp_ps[:, 0] = f['events/ps'][:]


        mmp_ts = np.expand_dims(np.array(f['events/ts'][:], dtype='float64'), axis=1)
        mmp_xy = np.stack((f['events/xs'][:], f['events/ys'][:])).transpose().astype('int16')
        mmp_ps = np.expand_dims(np.array(f['events/ps'][:], dtype='uint8'), axis=1)

        np.save(ts_path, mmp_ts)
        np.save(xy_path, mmp_xy)
        np.save(ps_path, mmp_ps)


        for i, data in enumerate(additional_data):
            if data == 'images':
                save_additional_data_as_mmap(f, mmap_pth, additional_data[data])
            elif data == 'flow':
                save_flow_as_mmap(f, mmap_pth, input_flow_path, additional_data[data])

        write_metadata(f, metadata_path)


if __name__ == "__main__":
    """
    Tool to convert this projects style hdf5 files to the memmap format used in some RPG projects
    """
    # 使用含事件的H5及光流图片生成memmap文件
    # for i in range(950):
    #     if i == 107 or i == 382:
    #         # missed data
    #         continue
    #     train_path = '/mnt/data/liuhaoyue/data/e2vid/ecoco_depthmaps_test/train/sequence_' + str(i).zfill(10) + '/events_only_trail_noslomo/events.h5'  # !
    #     train_flow_path = '/mnt/data/liuhaoyue/data/e2vid/ecoco_depthmaps_test/train/sequence_' + str(i).zfill(10) + '/flow'
    #     train_output_path = '/mnt/data/liuhaoyue/data/e2vid/ecoco_depthmaps_test/train_only_trail_memmap/sequence_' + str(i).zfill(10)  # !
    #     print(train_output_path)
    #     h5_to_memmap(train_path, train_flow_path, train_output_path)
    #
    # for i in range(950, 1001):
    #     val_path = '/mnt/data/liuhaoyue/data/e2vid/ecoco_depthmaps_test/validation/sequence_' + str(i).zfill(10) + '/events_only_trail_noslomo/events.h5'  # !
    #     val_flow_path = '/mnt/data/liuhaoyue/data/e2vid/ecoco_depthmaps_test/validation/sequence_' + str(i).zfill(10) + '/flow'
    #     val_output_path = '/mnt/data/liuhaoyue/data/e2vid/ecoco_depthmaps_test/val_only_trail_memmap/sequence_' + str(i).zfill(10)  # !
    #     print(val_output_path)
    #     h5_to_memmap(val_path, val_flow_path, val_output_path)

    # 生成文件路径txt
    train_data_memmap = 'E:/4000datasets/EventCamera/e2v_traindata/txt/v3/train_hog_memmap.txt'
    val_data_memmap = 'E:/4000datasets/EventCamera/e2v_traindata/txt/v3/val_hog_memmap.txt'

    train_txt = []
    for i in range(950):
        if i == 107 or i == 382:
            # missed data
            continue
        train_path = '/mnt/data/liuhaoyue/data/e2vid/ecoco_depthmaps_test/train_memmap/sequence_' + str(i).zfill(10) + '\n'
        train_txt.append(train_path)
    with open(train_data_memmap, 'w') as f:
        f.writelines(train_txt)

    val_txt = []
    for i in range(950, 1001):
        val_path = '/mnt/data/liuhaoyue/data/e2vid/ecoco_depthmaps_test/val_memmap/sequence_' + str(i).zfill(10) + '\n'
        val_txt.append(val_path)
    with open(val_data_memmap, 'w') as f:
        f.writelines(val_txt)
