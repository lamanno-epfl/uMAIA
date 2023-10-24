import numpy as np
import pandas as pd
import cv2
import zarr
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import numpyro
import anndata
import os


def extract_image_coordinates(coordinates, img_shape, values):
    """extract image from IBD files where coordinates are assigned to each pixel

    Args:
    ----
    coordinates: coordinate list from reader.coordinates
    img_shape: expected image shape
    values: values associated with coordinates
    """
    coords = np.array(coordinates)[:,[0,1]] - 1
    image = np.zeros(img_shape)
    for i, row in enumerate(coords[:len(values)]):
        image[row[1], row[0]] = values[i]
    return image


def filter_mz(df:pd.DataFrame, mz_list:list, threshold:float=0.005):
    """
    Remove mz ranges in the dataframe that correspond to signals representing known matrix compounds or other unwanted compounds.
    Removal is based on the value that is closest to the inputted query list, and is removed if the difference is greater than the provided threshold.
    
    Args:
    ----
    df: pd.DataFrame
        dataframe containing mz_estimated column for set of bins retrieved from acquisition
    mz_list: list or array
        list containing mz values that are desired to be filtered out
    threshold: float - default 0.005
        value indicating the maximum difference between theoretical and detected mz value to be considered as matched
        increasing this value will remove more molecules (and potentially false positives)

    Returns
    --------
    pd.DataFrame without undesired mz values
    """
    mz_list = np.array(mz_list).flatten()
    ix = np.argwhere(df.mz_estimated.apply(lambda x: np.min(np.abs(x - mz_list))).values >= threshold).flatten()
    return df.iloc[ix]


def filter_1stdecimal(df:pd.DataFrame, first_decimal_range:tuple=(1, 9)):
    """
    Remove values according to the first digit after the decimal in the mz value
    
    Args:
    ----
    df: pd.DataFrame
        Dataframe with mz_estimated column.
    first_decimal_range: tuple
        Tuple of size 2. First value indicates the values below that should be removed. Second value indicates the values above that should be removed
    
    Returns
    -------
    pd.DataFrame excluding undesired mz ranges
    
    """
    if first_decimal_range[1] > first_decimal_range[0]:
        ix_select = np.argwhere(
            (df.mz_estimated.apply(
                lambda x: np.floor((x * 10) % 10)).values >= first_decimal_range[0]) & (df.mz_estimated.apply(
                lambda x: np.floor((x * 10) % 10)).values <= first_decimal_range[1])
        ).flatten()
    else:
        ix_select = np.argwhere(
            (df.mz_estimated.apply(
                lambda x: np.floor((x * 10) % 10)).values >= first_decimal_range[0]) | (df.mz_estimated.apply(
                lambda x: np.floor((x * 10) % 10)).values <= first_decimal_range[1])
        ).flatten()
    return df.iloc[ix_select]


def extract_contours_from_binary(mask):
    """Given a binary mask, extract the largest contour"""
    cv_img = mask.astype(np.uint8)
    ret,thresh1 = cv2.threshold(cv_img,127,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(cv_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.squeeze(x) for x in contours if len(x) > 50][0]
    
    return contours


def read_files(files:list):
    df_list = []
    for file in files:
        # load peakcalled file
        df = pd.read_csv(file, index_col=0)
        df_list.append(df)
    df_list = np.array(df_list, dtype=object)
    return df_list


def read_images_masks(acquisitions:list,
                      path_images:str, path_masks:str = None,
                      gaussian_smoothing:bool = False, gaussian_sigma:float = 0.3,
                      log_transform:bool = True, epsilon:float = 0.0002):
    """ Reading images and masks saved for normalization step,
    also returns masks_list
    
    Args:
    ----
    acquistions: list
        list of strings containing the path to each section acquisition
        
    path_images: str
        path to zarr file saved from the molecular matching step
        
    path_masks: str
        path to mask files (apart from the mask.npy extension)
        
    gaussian_smoothing: bool - default: True
        boolean indicator of applying gaussian filter on raw data before normalization
    
    gaussian_sigma: float - default: 0.3
        value of gaussian variance for gaussian smoothing (ignored if gaussian_smoothing=False)
        
    log_transform: bool - default: True
        boolean indicator of whether you want to work with log-transformed data
        
    epsilon: float - default: 0.0002
        small number to make sure log transform does not return NaN
    """
    
    
    
    root = zarr.open(path_images, mode='rb')
    PATH_MZ = np.sort(list(root.group_keys()))
    
    
    if path_masks:
        masks_list = [np.load(os.path.join(path_masks, name, 'mask.npy')) for name in acquisitions]
    else:
        masks_list = [np.ones_like(root[PATH_MZ[0]][i_s][:]) for i_s in range(len(acquisitions))]
    
    mask_ix_list = [np.argwhere(x.flatten()).flatten() for x in masks_list]
    
    # initialize two variables for the data and the mask
    x = np.ones((np.max([len(np.argwhere(x.flatten()).flatten()) for x in masks_list]), len(masks_list), len(PATH_MZ))) 
    mask = np.zeros_like(x, dtype=bool)
    
    if log_transform==True:
        x = np.log(x + epsilon)
    
    
    for i_v, mz in tqdm(enumerate(PATH_MZ), desc="Loading Data..."): # for a single molecule:

        for i_s in range(len(acquisitions)): # for a single molecule across the sections
        
            try:
                image = root[mz][i_s][:]
                
                if gaussian_smoothing==True:
                    image = gaussian_filter(image, gaussian_sigma)
            except:
                continue
                
            if log_transform==True:
                if np.sum(image.astype(float)) == 0.0:
                    image = np.log(np.ones_like(image).astype(float) + epsilon)
                else:
                    image = np.log(np.nan_to_num(image).astype(float) + epsilon)

                
            # extract masked image
            img_masked = image.flatten()[mask_ix_list[i_s]]
            x[:len(mask_ix_list[i_s]), i_s, i_v] = img_masked
            mask[:len(mask_ix_list[i_s]), i_s, i_v] = True 

    if not path_masks:
        masks_list = np.ones_like(x)
            
    print('Data Loaded Successfully.')
    return x, mask, masks_list


def createSaveDirectory(path_save):
    """ Helper function to create a new directory for saved results
    
    Args:
    ----
    path_save:  str
    """
    # check that the directory does not already exist
    
    if not os.path.isdir(path_save):
        os.mkdir(path_save)
    else:
        print(f'Directory at {path_save} already exists')
     
        
def filterSparseImages(df_list, num_pixels=50, percentile=None, masks=None):
    df_list_2 = []
    for i, df in enumerate(df_list):
        
        if num_pixels:
            df = df[df.num_pixels > num_pixels]
        else:
            df = df[df.num_pixels > np.percentile(df.num_pixels.values, percentile)] # keep if there are more than a certain number of signals inside the tissue
        df_list_2.append(df)

    return np.array(df_list_2, dtype=object)


def to_zarr(PATH_SAVE:str, acquisitions:list, df_filter:pd.DataFrame, images_list:list):
    """ Function to save matched molecules into zarr file
    
    Args:
    ----
    PATH_SAVE: str
        Path to save the .zarr file
    
    acquisitions: list
        list of acquisition names
        
    df_filter: pd.DataFrame
        matched file for creating images
        
    images_list: list
        paths of anndata objects
        
    """
    
    # initialise dataset
    root = zarr.open(PATH_SAVE, mode='w')

    for s in tqdm(np.arange(0, len(acquisitions))): # iterate over sections

        # load the file in question
        images = anndata.read_h5ad(images_list[s]) 
        img_shape = images.uns['img_shape']
        # subset the df_match for the section
        df_match_sub = df_filter[df_filter.section_ix == s]
        # retrieve the complete set of observed molecules from section S
        mz_values = images.var.values.flatten()
        
        idx_lipid = [] # initialize list for indexes of identified compounds
        mz_values_ref = [] # mz values (as str)
        for i, mz in enumerate(df_match_sub.mz_estimated.values):
            # extract the image
            argmin = np.argmin(np.abs(mz_values - mz))
            if np.abs(mz_values - mz)[argmin] <= 0.005:
                idx_lipid.append(argmin)
                mz_values_ref.append(df_match_sub.mz_estimated_total.values[i])
        idx_lipid = np.array(idx_lipid).astype(int)
        images_lipid = images.X[:,idx_lipid] # subset the images to include only those of interest
        mz_values_ref = [f'{x:.4f}' for x in mz_values_ref] # corresponding mz to the subsetted images
        del images # remove images from memory


        # iterate over lipids
        for i, mz in enumerate(mz_values_ref):
            # create subfoloder if it doesn't already exist
            if not os.path.isdir(os.path.join(PATH_SAVE, mz)):
                os.mkdir(os.path.join(PATH_SAVE, mz))
            # select_image
            img = images_lipid[:,i]
            # reshape image
            img = img.reshape(img_shape)
            # save image to path
            root.create_dataset(f'{mz}/{s}', data=img)    


def save_svi(svi_result: numpyro.infer.svi.SVIRunResult,
             save_path: str):
    """ Helper function to save the results of MAIA normalization method
    
    Args:
    ----
    svi_result:  numpyro.infer.svi.SVIRunResult
        Results of MAIA SVI on data, contaning the parameters required to remove the batch effects
        
    save_path: str
        Path to save the parameters
    """
    
    weights = svi_result.params['weights_auto_loc']
    locs = svi_result.params['locs_auto_loc']
    scale1 = svi_result.params['scale1_auto_loc']
    sigma_v = svi_result.params['sigma_v_auto_loc']
    b_lambda = svi_result.params['b_lambda_auto_loc']
    b_gamma = svi_result.params['b_gamma_auto_loc']
    delta = svi_result.params['delta_auto_loc']
    sigma_s = svi_result.params['sigma_s_auto_loc']
    error = svi_result.params['error_auto_loc']
    delta_ = svi_result.params['delta_']
    loc0_delta = svi_result.params['loc0_delta']
    
    np.save(os.path.join(save_path, 'weights.npy'), weights)
    np.save(os.path.join(save_path, 'locs.npy'), locs)
    np.save(os.path.join(save_path, 'delta.npy'), delta)
    np.save(os.path.join(save_path, 'scale1.npy'), scale1)
    np.save(os.path.join(save_path, 'sigma_v.npy'), sigma_v)
    np.save(os.path.join(save_path, 'b_lambda.npy'), b_lambda)
    np.save(os.path.join(save_path, 'sigma_s.npy'), sigma_s)
    np.save(os.path.join(save_path, 'b_gamma.npy'), b_gamma)
    np.save(os.path.join(save_path, 'delta_.npy'), delta_)
    np.save(os.path.join(save_path, 'error.npy'), error)
    np.save(os.path.join(save_path, 'loc0_delta.npy'), loc0_delta)
    