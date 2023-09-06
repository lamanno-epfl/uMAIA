import numpy as np
import pandas as pd
import cv2

def extract_image_coordinates(coordinates, img_shape, values):
    """extract image from IBD files where coordinates are assigned to each pixel

    Args
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
    
    Args
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
    
    Args
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
    df_list = np.array(df_list)
    return df_list