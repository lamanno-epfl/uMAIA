import numpy as np
import pandas as pd
import cv2
import zarr
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import numpyro
import anndata
import os
from skimage.measure import regionprops




def load_ann(PATH_ANN:str):
    """ Function to save matched molecules into zarr file
    
    Args:
    ----
    PATH_ANN: str
        Path to the h5ad file
    """

    # load the file in question
    h5ad = anndata.read_h5ad(PATH_ANN, backed='r') 

    return h5ad

def visualise_image_h5ad(h5ad_file, mz):

    # retrieve the complete set of observed molecules from section S
    mz_values = h5ad_file.var.values.flatten()
    img_shape = h5ad_file.uns['img_shape']
    

    # extract the image
    argmin = np.argmin(np.abs(mz_values - mz))
    if not np.abs(mz_values - mz)[argmin] <= 0.005:
        print(f'Closest match found {np.abs(mz_values - mz)}')
        
    img = h5ad_file.X[:,argmin] # subset the images to include only those of interest
    # reshape image
    img = img.reshape(img_shape)

    return img

def singlecell_intensities(image_h5ad, label_image, df_ranges):
    """Extract sum of intensities within regions indicated by label map
    Args
    ----
    h5ad_file: h5ad object containing acquisition images
    label_image: 2D Array
        dimension the same as smz.img_shape. Represents individual entities in the image by integer values
    df_ranges: pd.DataFrame
        output from build_ranges_df
    """
    # Extract regions object, each entry a cell
    list_regs = regionprops(label_image)
    img_shape = image_h5ad.uns['img_shape']
    
    
    mz_values = image_h5ad.var.values.flatten()
    
    df_singlecells = pd.DataFrame()
    for i, (name, selected_row) in tqdm.tqdm(enumerate(df_ranges.iterrows()), total=len(df_ranges)):
        img_listsum = []
        
        img = visualise_image_h5ad(image_h5ad, selected_row.mz_estimated)
        
        # iterate over cells
        for cell in list_regs:
            sum_ = 0
            coords = cell.coords
            for coord in coords:
                sum_ += img[coord[0], coord[1]]
            img_listsum.append(sum_)
        df_singlecells[selected_row.mz_estimated] = img_listsum
    # set row labels according to file labels
    df_singlecells.index = [x.label for x in list_regs]
    return df_singlecells


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

#####################################################################################
# aaf edited function:
    # tp np array default behavior when dfs in df list equal number of rows
    # 
#####################################################################################
# def read_files(files:list):
#     df_list = []
#     for file in files:
#         # load peakcalled file
#         df = pd.read_csv(file, index_col=0)
#         df_list.append(df)
#     df_list = np.array(df_list, dtype=object)
#     return df_list

# type(df_list) >> numpy.ndarray
# cada type(df_list[i]) >>  pandas.core.frame.DataFrame

def read_files(files: list) -> np.ndarray:
            # specify tipos annotation entra lista sale np.array 
    '''
        - This function was edited - 
        No changes in input or output so function is replaced
    '''
    df_list = []
    for file in files:
        df = pd.read_csv(file, index_col=0)
        # Forzar a DataFrame si no lo es (le da igual el issue es np.array def )
        df = pd.DataFrame(df)
        df_list.append(df) # 
    
    # construir manual para evitar estructura homogenea 2D!
    df_array = np.empty(len(df_list), dtype=object)
    for i, df in enumerate(df_list):
        df_array[i] = df
    
    # return df_list
    return df_array

#########################################################################################################
    # end aaf
#########################################################################################################

def read_images_masks(acquisitions:list,
                      path_images:str, mask_list:list,
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
        
    mask_list: list
        list of numpy arrays corresponding to images
        
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

    
    mask_ix_list = [np.argwhere(x.flatten()).flatten() for x in mask_list]
    
    # initialize two variables for the data and the mask
    x = np.zeros((np.max([len(np.argwhere(x.flatten()).flatten()) for x in mask_list]), len(mask_list), len(PATH_MZ))) 
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

            
    print('Data Loaded Successfully.')

    # correct for empty images
    masks_sum = np.sum(mask, axis=2).astype(bool)
    mask[:,:,:] = masks_sum[:,:,None]
    return x, mask, mask_list


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

#####################################################################################
# aaf edited function:
    # tp np array default behavior when dfs in df list equal number of rows
    # 
#####################################################################################
        
# def filterSparseImages(df_list, num_pixels=50, percentile=None, masks=None):
#     df_list_2 = []
#     for i, df in enumerate(df_list):
        
#         if num_pixels:
#             df = df[df.num_pixels > num_pixels]
#         else:
#             df = df[df.num_pixels > np.percentile(df.num_pixels.values, percentile)] # keep if there are more than a certain number of signals inside the tissue
#         df_list_2.append(df)

#     return np.array(df_list_2, dtype=object)

def filterSparseImages(df_list, num_pixels=50, percentile=None, masks=None):
    '''
        - This function was edited - 
        No changes in input or output so function is replaced
    '''

    df_list_2 = []
    for i, df in enumerate(df_list):
        
        if num_pixels:
            df = df[df.num_pixels > num_pixels]
        else:
            df = df[df.num_pixels > np.percentile(df.num_pixels.values, percentile)] # keep if there are more than a certain number of signals inside the tissue
        df_list_2.append(df)

    # return np.array(df_list_2, dtype=object) # EVITA ESTO!
    # construir manual para evitar estructura homogenea 2D!
    df_array = np.empty(len(df_list_2), dtype=object)
    for i, df in enumerate(df_list_2):
        df_array[i] = df
        
    return df_array

#########################################################################################################
    # end aaf
#########################################################################################################


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
        mz_values_ref = [f'{x:.6f}' for x in mz_values_ref] # corresponding mz to the subsetted images
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


def place_image(masks_list, tranformed_values, v, s, epsilon):
    img = np.zeros(masks_list[s].shape).flatten()
    #img[mask_list[s].flatten()] = np.exp(tranformed_values[:np.sum(mask_list[s]),s,v]) - epsilon
    img[masks_list[s].flatten()] = tranformed_values[:np.sum(masks_list[s]),s,v]
    img[~masks_list[s].flatten().astype(bool)] = epsilon
    return img.reshape(masks_list[s].shape)
    

def to_zarr_normalised(PATH_originalZarr, PATH_normZarr, acquisitions, x_tran, masks_2D, small_num):
    root = zarr.open(PATH_originalZarr, mode='rb')
    mz_list = np.array(list(root.group_keys())) 
    root = zarr.open(PATH_normZarr, mode='w')
    
    for i_v, v in enumerate(mz_list):
        for s in np.arange(0, len(acquisitions)): # iterate over sections,
    
            img = place_image(masks_2D, x_tran, i_v, s, np.log(small_num))
            # img = np.exp(img) - small_num
            root.create_dataset(f'{v}/{s}', data=img)



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
    
