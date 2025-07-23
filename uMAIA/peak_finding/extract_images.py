import uMAIA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import *
import pandas as pd
import tqdm
import shutil
import concurrent
import anndata

import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description="Run uMAIA processing script.")
parser.add_argument("--path_data", type=str, required=True, help="Full path to data folder")
parser.add_argument("--name", type=str, required=True, help="Sample name")
parser.add_argument("--mzresolution", type=float, required=False,default=0.0001, help="mz resolution to load data")
parser.add_argument("--threshold_count", type=float, required=False,default=8, help="threshold count for watershed limit")
parser.add_argument("--approximate_interval", type=float, required=False,default=0.1, help="approximate interval for m/z chunk to be processed")
parser.add_argument("--smoothing", type=float, required=False,default=1., help="gaussian kernel smoothing on frequency representation")
parser.add_argument("--spectrum_range", type=tuple, required=False,default=(400.,1200.), help="mz range to process")
args = parser.parse_args()


# path_data = '/data/CERT_project/data/'
# path_save = '/data/CERT_project/data/'
# name = '20240624_GDA990_BRAINc05_419X340_25UM_Att30'
# Assign to variables
path_data = args.path_data
path_save = args.path_data
mz_resolution = args.mzresolution
threshold_count = args.threshold_count
approximate_interval = args.approximate_interval
smoothing = args.smoothing
spectrum_range = args.spectrum_range
name = args.name


def save_images(smz, df:pd.DataFrame, filepath:str, mz_list:list=None,  normalization='tic', clip=100, img_shape=None, coordinates=None):
    """
    Extract images from the smzObj according to extracted ranges defined in df and save into AnnData object (.h5ad) format
    
    Args
    ----
    smz: smzObj
    df: pd.DataFrame
        dataframe containing extracted ranges from smz. Relevant columns are ['min', 'max', 'mz_estimated']
    filepath: str
        path to save .h5ad file
    mz_list: list[float], optional
        list containing m/z values of molecules desired for imaging. Approximations are made if the mz value is not present in df.
        no value provided will save all molecules present in df by default
    normalization: str or float, default='tic'
        indication of normalization form. when 'tic' is provided, images are normalized by the 
    clip: int, optional
        percentile to clip images
        default, no clipping
    """
    
    if normalization == 'tic':
        tic_pixels = np.array(smz.S.sum(axis=1)).flatten()
    elif isinstance(normalization, float):
        # retrieve the bin of the mz value
        ix = np.argmin(df.mz_estimated - normalization)
        selected_row = df.iloc[ix]
        r = selected_row['min'], selected_row['max']
        mz_select_ = (smz.mz_vals > r[0]) & (smz.mz_vals < r[1])
        S_select = smz.S[:,mz_select_].toarray()
        tic_pixels = np.sum(S_select, axis=1)
        tic_pixels = np.array(tic_pixels).flatten()
    else:
        pass

    if img_shape is None:
        img_shape = smz.img_shape

    if mz_list is None:
        mz_list = df.mz_estimated.values
        
    X = anndata.AnnData(X=np.zeros([ np.multiply(*img_shape), len(mz_list)]), var=mz_list)

    for i, r in tqdm.tqdm(enumerate(mz_list), total=len(mz_list)):

        ix = np.argmin(np.abs(df.mz_estimated - r))
        selected_row = df.iloc[ix]
        r = selected_row['min'], selected_row['max']
        mz_select_ = (smz.mz_vals > r[0]) & (smz.mz_vals < r[1])
        S_select = smz.S[:,mz_select_].toarray()
        mz = smz.mz_vals[mz_select_]

        img = np.sum(S_select, axis=1)
        if normalization != None:
            img = img / tic_pixels

        # clip to nth percentile
        percentile = np.percentile(img, clip)
        img = np.clip(img, 0, percentile)

        if img.shape[0] < np.multiply(*img_shape):
            img = uMAIA.ut.tools.extract_image_coordinates(coordinates, img_shape, img)
        else:
            img = img[:np.multiply(*img_shape)].reshape(img_shape)
        

        # insert into X
        X.X[:,i] = img.flatten()
    # save into anndata h5ad format
    X.var.columns = X.var.columns.astype(str) # convert variables to strings
    X.uns["img_shape"] = list(img_shape)
    X.write(filepath, compression='gzip')

warnings.filterwarnings("ignore")


# create directories
uMAIA.ut.tools.createSaveDirectory(path_save)
uMAIA.ut.tools.createSaveDirectory(os.path.join(path_save, name))

smz = uMAIA.ut.SmzMLobj(f'{os.path.join(path_data,name,name)}.IBD',
                           f'{os.path.join(path_data,name,name)}.imzml',
                           mz_resolution=mz_resolution)

smz.load(load_unique_mz=True)
smz.S


# mz_resolution = 0.0001
# threshold_count = 8.
# approximate_interval =0.1
# smoothing = 1. #2.5

print('running uMAIA...')
uMAIA.pf.run(directory_path=os.path.join(path_save, name),
            smz=smz, 
            spectrum_range=spectrum_range,
            threshold_count=threshold_count, 
            approximate_interval=approximate_interval,
            smoothing=smoothing,
            parallelize=True,
            saveimages=False)


df_ranges = pd.read_csv(os.path.join(path_save, name, 'ranges.csv'), index_col=0)
path_save = os.path.join(path_save, name, 'images.h5ad')

print(path_save)
df_filtered = df_ranges[df_ranges.num_pixels > 100]
print(df_filtered.shape)
save_images(smz, df=df_filtered, filepath=path_save,img_shape=smz.img_shape, coordinates=smz.reader.coordinates)

print('completed')