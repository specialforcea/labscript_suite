import h5py
import numpy as np

def get_raw_images(fpath):
    with h5py.File(fpath) as h5_file:
        if  '/data/imagesYZ_2_Flea3' in h5_file:
            image_group = 'imagesYZ_2_Flea3'
        if '/data/imagesXY_1_Flea3' in h5_file:
            image_group = 'imagesXY_1_Flea3'
        if False: #'/data/imagesXY_2_Flea3' in h5_file:
            image_group = 'imagesXY_2_Flea3'
        if '/data/imagesYZ_1_Flea3' in h5_file:
            image_group = 'imagesYZ_1_Flea3'
        atoms = np.array(h5_file['data'][image_group]['Raw'])[0]
        probe = np.array(h5_file['data'][image_group]['Raw'])[1]
        bckg = np.array(h5_file['data'][image_group]['Raw'])[2]
        return atoms, probe, bckg