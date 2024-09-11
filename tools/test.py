import numpy as np

# read numpy array from file
data = np.load('datasets/fluencybank/data_sed/sed_inputs_A1_train.npy')
label =np.load('datasets/fluencybank/data_sed/sed_outputs_A1_train.npy')

print(data.shape)
print(label.shape)