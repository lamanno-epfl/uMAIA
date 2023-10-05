import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import *
import pandas as pd
import tqdm
import shutil
import concurrent
from ..utils.tools import extract_image_coordinates
import anndata
from .peak_finder import PeakFinder
import os
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

def run(directory_path:str, smz, spectrum_range:tuple, threshold_count:int=10, approximate_interval:float=1.0, 
smoothing:float=3.,
parallelize=True, saveimages=True, mz_resolution=0.0001):
    """
    Runs peakcaller and outputs corresponding csv file with metadata for individual peaks
    
    Args
    ----
    file_path: str
        path indicating where csv output should be stored
    smz: smzObj
    spectrum_range:tuple
        tuple containing minimum and maximum values for spectrum scan
    threshold_count:int - default 10
        value to threshold noise signal frequencies
    approximate_interval:float - default 1.0
        value to partition the mz space for processing
    smoothing:float - default 3.0
        sigma value of gaussian smoothing kernel for frequency values
    
    Returns
    -------
    pd.DataFrame containing min and max values of individual bins, and additional columns with metadata for the acquisition
    
    """
    peakcall(file_path=directory_path,smz=smz, spectrum_range=spectrum_range, threshold_count=threshold_count,
    approximate_interval=approximate_interval, smoothing=smoothing,
    mz_resolution=mz_resolution,
    parallelize=parallelize) # 1050, 1100
    print('Creating dataframe...')
    df_ranges = create_dataframe(smz=smz, file_path=directory_path)
    print('DataFrame created successfully')
    # save dataframe to file_path
    df_ranges.to_csv(os.path.join(directory_path, 'ranges.csv')) # save file
    shutil.rmtree(os.path.join(directory_path, 'ranges'))
    print('Removing temporary storage')
    
    # save image files as 
    if saveimages:
        print('Saving images...')
        save_images(smz=smz, df=df_ranges,
            filepath=os.path.join(directory_path, 'images.h5ad'),
            clip=100, coordinates=smz.reader.coordinates)

    print('Complete')



def run_peakfinder_parallel(S, mz_vals, mz_range, threshold_count=10, smoothing=2.0, mz_resolution=0.0001):
    """peakcaller for a single instance"""
    PF = PeakFinder(S=S.copy(), mz_vals=mz_vals.copy(), mz_range=mz_range, threshold_count=threshold_count, smoothing=smoothing, mz_resolution=mz_resolution)
    success = PF.process()
    if success:
        PF.fit()  
    return PF.ranges



def retrieve_ranges(S, mz_vals, spectrum_range, threshold_count, approximate_interval=1.):
    """extract ranges suitable for peakcalling, such that peaks are encapsulated within the range"""
    
    S_freq = np.array(S.astype(bool).astype(int).sum(axis=0)).flatten()
    ix_potential = np.argwhere(S_freq < threshold_count).flatten()
    ideal_points = np.append(np.arange(spectrum_range[0], spectrum_range[1], approximate_interval),
                             spectrum_range[1])
    
    mz_potential = mz_vals[ix_potential]
    # check if data have been pre-processed
    if len(mz_potential) == 0:
        ranges = np.column_stack([ideal_points, ideal_points + approximate_interval])
    else:
    # retrieve the most appropriate mz value given the thresholds
        mz_select = [find_nearest(mz_potential, x) for x in ideal_points]
        mz_select = np.unique(mz_select)
        # create a list of list for the ranges
        ranges = np.column_stack([mz_select[:-1], mz_select[1:]])
    
    return ranges

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def peakcall(file_path, smz, spectrum_range:tuple=(400,1200), threshold_count=5, approximate_interval=1.0, smoothing=2.,for_vis=False, parallelize=True, mz_resolution=0.0001):
    """peakcall returns all bins for designated spectrum in acquisition
    
    Args
    ----
    smz: SmzObject
    spectrum_range: tuple
        mz range one wishes to extract binsn from
    threshold_count: int
        the minimum frequency to consider for GMM fitting
        
    Returns
    -------
    ranges: np.ndarray
        contains sets of mz bins, where each row is a separate bin and columns are min, max of range respectively
    """

    mz_select = (smz.mz_vals >= spectrum_range[0]) & (smz.mz_vals <= spectrum_range[1])
    mz_vals = smz.mz_vals[mz_select]
    S = smz.S[:,mz_select]
    
    print('Partioning m/z space and beginning peakcalling...')
    ranges = retrieve_ranges(S, mz_vals, spectrum_range=spectrum_range, threshold_count=threshold_count, approximate_interval=approximate_interval)
    ranges = ranges.reshape(-1,2)
    # print(sys.getsizeof(ranges))
    if not for_vis:
        os.mkdir(os.path.join(file_path, 'ranges'))

    if parallelize:
        # iterate over sets of ranges so that the input for argument list is not too big. We want to partition in to jobs of 10 at a time
        num_iter = int(np.ceil(len(ranges) / 5))
        results_list = []
        for i in tqdm.tqdm(range(num_iter), total=num_iter):
            
            subset = i * 5
            if subset + 5 > len(ranges):
                ranges_sub = ranges[subset:]
            else:
                ranges_sub = ranges[subset:subset+5]

            data = []
            for range_ in ranges_sub:
                
                ix = np.argwhere((mz_vals > range_[0]) & (mz_vals < range_[1])).flatten()
                d = [S[:,ix], mz_vals[ix], range_, threshold_count, smoothing,mz_resolution]
                data.append(d)   

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(run_peakfinder_parallel, *(zip(*data)))) # type: ignore
                results = [r for r in results if len(r) > 0]
                results = np.concatenate(results)
                


            if for_vis:
                results_list.append(results)
            else:
                np.savetxt(os.path.join(file_path, 'ranges',f'{i}.gz'), results)

    else:
        for iter_, mz_range in tqdm.tqdm(enumerate(ranges), total=len(ranges)):
            results = run_peakfinder_parallel(S=S, mz_vals=mz_vals,
                                                mz_range=mz_range, 
                                                threshold_count=threshold_count,
                                                smoothing=smoothing,
                                                mz_resolution=mz_resolution
                                                )
            try:
                results = np.concatenate(results).reshape(-1,2)
                np.savetxt(os.path.join(file_path, 'ranges',f'{iter_}.gz'), results)
            except:
                continue
        
    if for_vis:

        S_compressed, data_mz = np.histogram(np.array(S.todense()).astype(bool).sum(axis=0).astype(bool) * mz_vals,
                                  bins=np.arange(spectrum_range[0],
                                                 spectrum_range[1],
                                                 mz_resolution * 7))
        S_smooth = gaussian_filter1d(S_compressed, sigma=smoothing)
        return results, S_smooth



def build_ranges_df(smz, ranges):
    """Builds dataframe of bins including metadata such as intensity, bin size, number of pixels detected, and purity of bin
    
    Args
    ----
    smz: SmzObj
    ranges: np.ndarray
        shape (n, 2). For n bins. First column indicates minimum bin value, second column maximum bin values
    """

    df_ranges = pd.DataFrame(ranges, columns=['min', 'max'])
    pixel_max_hits = []
    num_pixels = []
    pixel_percent_1hit = []
    median_intensity_list = []
    percent_in_matrix_list = []
    expected_mz = []
    concentration = []


    tic_pixels = smz.S.sum(axis=1)
    for index, row in tqdm.tqdm(df_ranges.iterrows(), total=len(df_ranges)):
        selected_index = (smz.mz_vals >= row['min']) & (smz.mz_vals <= row['max'])
        S_selected = smz.S[:,selected_index].astype(bool).astype(int).sum(axis=1)
        try:
            mz_approx = smz.mz_vals[selected_index][np.argmax(smz.S[:,selected_index].astype(bool).astype(int).sum(axis=0))]
        except:
            mz_approx = np.nan
        percent_1_hit = np.sum(S_selected == 1.) / np.sum(S_selected >= 1.) * 100
        try:
            S_tmp = smz.S[:,selected_index].sum(axis=1) / tic_pixels
            median_intensity = np.nanpercentile(S_tmp.data, 50)
            c_ = np.nansum(S_tmp)
        except:
            median_intensity = 0
            c_ = 0

        pixel_percent_1hit.append(percent_1_hit)
        pixel_max_hits.append(np.max(S_selected))
        num_pixels.append(np.count_nonzero(S_selected))
        median_intensity_list.append(median_intensity)
        expected_mz.append(mz_approx)
        concentration.append(c_)

    df_ranges['pixel_max_hits'] = pixel_max_hits
    df_ranges['num_pixels'] = num_pixels
    df_ranges['percent_1_hit'] = pixel_percent_1hit
    df_ranges['concentration'] = concentration
    df_ranges['median_intensity'] = median_intensity_list
    df_ranges['mz_estimated'] = expected_mz
    df_ranges['difference'] = df_ranges['max'] - df_ranges['min']

    # remove rows where m/z estimated is the same. keep the row with the largest bin size
    # df_ranges = df_ranges.loc[df_ranges.groupby('mz_estimated')['difference'].idxmax()]

    return df_ranges

    
def create_dataframe(smz, file_path):
    """Retrieves numpy arrays from temp storage and builds dataframe for ranges"""
    files = os.listdir(os.path.join(file_path, 'ranges'))
    ranges = []
    for file in files:
        r = np.loadtxt(os.path.join(file_path, 'ranges', file))
        ranges.append(r)

    ranges = np.vstack(ranges)
    
    df_ranges = build_ranges_df(smz, ranges).dropna()
    
    return df_ranges


# generate some images based on suggested mz value
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
            img = extract_image_coordinates(coordinates, img_shape, img)
        else:
            img = img[:np.multiply(*img_shape)].reshape(img_shape)
        

        # insert into X
        X.X[:,i] = img.flatten()
    # save into anndata h5ad format
    X.var.columns = X.var.columns.astype(str) # convert variables to strings
    X.uns["img_shape"] = list(img_shape)
    X.write(filepath, compression='gzip')
