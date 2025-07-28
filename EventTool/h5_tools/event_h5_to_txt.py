#!/usr/bin/env python

import sys
from tqdm import tqdm

import h5py
import hdf5plugin
import numpy as np

import matplotlib.pyplot as plt

# Set file name
h5name = "$input";
filename = "$output"

# Set np_array object chunk scale
chunk_scale = 1000000

# Read h5 file
with h5py.File(h5name, "r") as h5f:
    # Get a h5py dataset object
    data = h5f['events']
    # Show list of h5f files
    print(list(h5f.keys()))
# Show list of event data
print(list(data.keys()))

num_events = data['p'].shape[0]
# Set time unit us
ts = np.divide(data['t'],1e6)

# Write txt header
with open(filename, 'w') as f:
    f.write("# events\n")
    f.write("# timestamp x y polarity\n")

# Write event data on txt
stride = 100000; # event data stride
arrform='\t'.join(['%f'] + ['%d']*3)
# first iteration to last - 1
for i in tqdm(range(num_events//chunk_scale)):
    st = i * chunk_scale
    ed = (i+1)*chunk_scale-1        
    data_arr = np.column_stack((ts[st:ed:stride],data['x'][st:ed:stride], data['y'][st:ed:stride], data['p'][st:ed:stride]))
    with open(filename, 'a') as f:
        np.savetxt(f, data_arr, fmt=arrform)
# last iteration
st = num_events//chunk_scale * chunk_scale
ed = num_events-1;
data_arr = np.column_stack((ts[st:ed:stride],data['x'][st:ed:stride], data['y'][st:ed:stride], data['p'][st:ed:stride]))
with open(filename, 'a') as f:
    np.savetxt(f, data_arr, fmt=arrform)
# Done
sys.exit("File writing done")

