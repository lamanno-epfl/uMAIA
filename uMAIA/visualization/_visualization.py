import matplotlib.pyplot as plt
import numpy as np
from ..utils.tools import extract_image_coordinates
from ..peak_finding import PeakFinder
import pandas as pd
import numpyro
import scipy.stats as stats
import anndata
import tqdm
import zarr
import os

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)



def plot_freqmz(smz, mzrange:tuple, figsize:tuple=(30,10), ylim:int=1000, threshold_count:int=None, smoothing:float=2.0, bins:bool=False, approximate_interval=0.5, metadata=None,
                parallelize=False,mz_resolution=0.0005):
    """
    Function to plot frequency spectrum for a given mz range. Parameter indicating whether peakcaller should be run on the sample
    """

    fig = plt.figure(figsize=figsize)

    selected_mz = (smz.mz_vals > mzrange[0]) & (smz.mz_vals < mzrange[1])

    mz = smz.mz_vals[selected_mz]
    spikes, mz = np.histogram(np.array(smz.S[:,selected_mz].todense()).astype(bool).sum(axis=0).astype(bool) * mz,
        bins=np.arange(mzrange[0],mzrange[1],smz.mz_resolution * 7))
    
   #spikes = smz.S[:,selected_mz].astype(bool).astype(int).sum(axis=0)

    # plt.plot(mz[:-1],spikes, alpha=0.2)
    #plt.plot(mz,np.array(spikes).flatten(), alpha=0.2)
    
    if threshold_count:
        plt.hlines(threshold_count, mzrange[0], mzrange[1], color='k',linestyles='--')
      
    if bins == True:
        PF = PeakFinder(smz.S, smz.mz_vals, mz_range=mzrange, threshold_count=threshold_count, smoothing=smoothing, mz_resolution=mz_resolution)
        PF.process()
        PF.fit()
        cmap = get_cmap(len(PF.ranges))

        for i, r in enumerate(PF.ranges):
            plt.hlines(-20, r[0], r[1],linewidth=5, color=cmap(i), alpha=0.5)        
        plt.plot(PF.data_mz, PF.S_smooth, color='darkblue', alpha=0.7)
        plt.plot(PF.data_mz, PF.S_compressed, alpha=0.7)

    plt.ylim([-50, ylim])

    return PF

# check of the images retrieved from here. useful to check for reference compounds
def visualize_ranges_test(smz, PF, clip=100, img_shape=None):
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
    
    tic_pixels = np.array(smz.S.sum(axis=1)).flatten()
    
    if img_shape is None:
        img_shape = smz.img_shape

    image_list = []
    for i, r in tqdm.tqdm(enumerate(PF.ranges), total=len(PF.ranges)):

        mz_select_ = (smz.mz_vals > r[0]) & (smz.mz_vals < r[1])
        S_select = smz.S[:,mz_select_].toarray()
        mz = smz.mz_vals[mz_select_]

        img = np.sum(S_select, axis=1)
        # tic normalization
        img = img / tic_pixels

        # clip to nth percentile
        percentile = np.percentile(img, clip)
        img = np.clip(img, 0, percentile)

        if img.shape[0] < np.multiply(*img_shape):
            img = extract_image_coordinates(smz.reader.coordinates, img_shape, img)
        else:
            img = img[:np.multiply(*img_shape)].reshape(img_shape)
            
        image_list.append(img)
    return image_list



def image_mz(smz, df:pd.DataFrame, mz_list:list, figsize:tuple=(15,15), ylim=None, cmap='inferno', img_shape=None, limit=0.01, normalize='tic', clip=99):
    """
    Image molecule in the dataset with the estimated m/z closest to the queried mz.
    Visualize the spectrum beside the image including the bin that was chosen

    If df is passed None, then a small range around the provided mz values will be imaged
    The small range is specified by the 'limit' parameter
    """
    tic_pixels = np.array(smz.S.sum(axis=1)).flatten()
    if img_shape is None:
        img_shape = smz.img_shape
    
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(len(mz_list), 2)

    mz_list_images = []
    
    for i, r in enumerate(mz_list):
        ax = plt.subplot(gs[i,0])
        ax_spectrum = plt.subplot(gs[i,1])
        
        if df is not None:
            ix = np.argmin(np.abs(df.mz_estimated - r))
            selected_row = df.iloc[ix]
            r = selected_row['min'], selected_row['max']
        else:
            r = (r - limit, r+limit)
        mz_select_ = (smz.mz_vals >= r[0]) & (smz.mz_vals <= r[1])
        S_select = smz.S[:,mz_select_].toarray()
        mz = smz.mz_vals[mz_select_]

        img = np.sum(S_select, axis=1)
        if normalize == 'tic':
            img = img / tic_pixels
        percentile = np.percentile(img, clip)
        img = np.clip(img, 0, percentile)

        if img.shape[0] < np.multiply(*img_shape):
            img = extract_image_coordinates(smz.reader.coordinates, img_shape, img)
        else:
            img = img[:np.multiply(*img_shape)].reshape(img_shape)
        ax.imshow(img, interpolation='none', cmap=cmap)
        ax.axis('off')
        ax.set_title(f'm/z estimated: {np.mean(r)}')

        # append image to list
        mz_list_images.append(img)


        ## plot ax_spectrum
        mz_select_ = (smz.mz_vals > r[0] - limit - 0.01) & (smz.mz_vals < r[1] + limit + 0.01)
        S_select = smz.S[:,mz_select_].toarray()
        mz = smz.mz_vals[mz_select_]

        ax_spectrum.plot(mz, S_select.astype(bool).astype(int).sum(axis=0), color='lightblue')
        ax_spectrum.hlines(-10, r[0], r[1], color='salmon', linewidth=10, alpha=0.7)
        ax_spectrum.set_xlabel('mz')
        ax_spectrum.set_ylabel('frequency')
        ax_spectrum.text(selected_row.mz_estimated, -10, str(selected_row.percent_1_hit)[:5], horizontalalignment='center')
        if ylim is not None:
            ax_spectrum.set_ylim([-15, ylim])
    return fig, mz_list_images

def plot_freqintensity(smz, df:pd.DataFrame, mzrange_list:list=[], figsize:tuple=(30,30), ylim_mz:int=250, ylim_intensity:int=500, threshold_count=5):
    """
    Function to plot frequency spectrum alongside intensity spectrum for specific ranges.
    
    Args
    ----

    smz: SmzObj
    df: pd.DataFrame
        Output from peakcall()
    mzrange_list: List[List]
        List of lists containing specified bins for plotting
    figsize: Tuple
        size of outputted plot
    ylim_mz: int
        ylimit for freq-mz plot
    ylim_intensity: int
        ylimit for intensity-mz plot
    """
    
    cmap = get_cmap(20)
    num_rows = len(mzrange_list)
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(num_rows,2)

    for iter_ in range(num_rows):

        ax = plt.subplot(gs[iter_ * 2])
        ax1 = plt.subplot(gs[iter_*2+1])

        if iter_ == 0:
            ax.set_title('freq-mz')
            ax1.set_title('intensity-mz')

        # specified range from GMM
        selected_mz = (smz.mz_vals > mzrange_list[iter_][0]) & (smz.mz_vals < mzrange_list[iter_][1])
        mz = smz.mz_vals[selected_mz]
        spikes = smz.S[:,selected_mz].astype(bool).astype(int).sum(axis=0)
        # spikes = smz.S[:,selected_mz].sum(axis=0)
        image = smz.S[:,selected_mz].sum(axis=1)

        
        # plot bins
        for i, (name, row) in enumerate(df[(df['min'] >= mzrange_list[iter_][0]) & (df['max'] <= mzrange_list[iter_][1])].iterrows()):
            ax.hlines(-5, row['min'], row['max'], color=cmap(i), linewidth=10, alpha=0.7)
            ax1.hlines(-5, row['min'], row['max'], color=cmap(i), linewidth=10, alpha=0.7)
            ax.text(row['min'], -30, str(row.percent_1_hit)[:5], rotation=90)

        ax.plot(mz,np.array(spikes).flatten(), alpha=0.2)
        ax.hlines(threshold_count, mzrange_list[iter_][0], mzrange_list[iter_][1],color='k', linestyles='--')
        ax.set_ylim([-40, ylim_mz])
        # plot intensity
        intensity = smz.S[:,selected_mz].sum(axis=0)
        ax1.plot(mz, np.array(intensity).flatten(), alpha=0.2)
        ax1.set_ylim([-20,ylim_intensity])
        ax.set_ylabel('frequency')
        ax1.set_ylabel('intensity')


def counts(df_list, xticks=None, fig=None):
    if fig is None:
        fig = plt.figure()
    plt.bar(np.arange(len(df_list)), [x.shape[0] for x in df_list])
    if xticks is not None:
        plt.xticks(np.arange(len(df_list)), xticks, rotation=90)
    plt.ylabel('counts')
    plt.show()
    
    
    
def plot_intensity(x: np.ndarray,
                   masks: np.ndarray,
                   v: int,
                   mz_val: float):
    
    """ Function to plot the intensity of one molecule across different sections
    
    Args:
    ----
    x: np.ndarray
        MALDI-MSI data
    
    mask: np.ndarray
        corresponding masks for MALDI-MSI data
        
    v: int
        number of molecule to plot its intensites across sections
        
    mz_val: float
        correponding value of mass-to-charge ratio (m/z) for molecule v
        
    """
    
    fig = plt.figure(None,(10,10), dpi=100)
    n_row = 5
    n_col = int(np.ceil(x.shape[1] / n_row))
    gs = plt.GridSpec(n_row, n_col)
    
    cm = plt.cm.get_cmap('bwr')
    xmin, xmax = np.percentile(x[:,:,v][masks[:,:,v]], (0.1, 99.90))
    xnew = np.linspace(xmin, xmax, 50)

    for i, s in enumerate(range(x.shape[1])):
        plt.subplot(gs[i])
        n, bins, patches = plt.hist(x[:,s,v][masks[:,s,v]], bins=xnew,
                                    color="gray", alpha=0.6, density=True, edgecolor='w')
        plt.ylim([0,2])

        if i == 0:
            plt.ylabel('density')
        plt.xlabel('intensity')
        
    if mz_val:
        plt.suptitle(mz_val)
        
    plt.show()
    
    
def normalized_hist(x_MAIA: np.ndarray, x: np.ndarray, 
                    mask: np.ndarray,
                    mask_2D_list: list,
                    svi_result,
                    zarr_path: str,
                    mz_val: float = None,
                    v: int = None,
                    covariates: list= None,
                    epsilon: float = 0.0002,
                    figsize=(15,80)
                    ):
    """ Function to plot the normalized vs raw data histogram and the fitted model
    
    Args:
    ----
    x_tran: np.ndarray
        Normalized MALDI-MSI data
    
    x: np.ndarray
        Raw MALDI-MSI data
        
    mask: np.ndarray
        corresponding masks for MALDI-MSI data
        
    svi_result: numpyro.infer.svi.SVIRunResult
        SVI results
        
    v: int
        number of molecule to plot its intensites across sections
        
    n_cov: list
        list indicating which sections should be considered covariates
        
    epsilon: float - default: 0.0002
        small number to make sure log transform does not return NaN
        
    mz_val: float
        correponding value of mass-to-charge ratio (m/z) for molecule v
    """
    if isinstance(svi_result, str):
        print('using saved values')
        weights = np.load(os.path.join(svi_result, 'weights.npy'))
        locs = np.load(os.path.join(svi_result, 'locs.npy'))
        scale1 = np.load(os.path.join(svi_result, 'scale1.npy'))
        sigma_v = np.load(os.path.join(svi_result, 'sigma_v.npy'))
        b_lambda = np.load(os.path.join(svi_result, 'b_lambda.npy'))
        b_gamma = np.load(os.path.join(svi_result, 'b_gamma.npy'))
        delta = np.load(os.path.join(svi_result, 'delta.npy'))
        sigma_s = np.load(os.path.join(svi_result, 'sigma_s.npy'))
        error = np.load(os.path.join(svi_result, 'error.npy'))
        delta_ = np.load(os.path.join(svi_result, 'delta_.npy'))
        locs_resampled = locs
        #loc0_delta = np.load(os.path.join(svi_result, 'loc0_delta.npy'))
    
    else:
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
        locs_resampled = locs
        #loc0_delta = svi_result.params['loc0_delta']
    
    N, S, V = x.shape

    if np.all(covariates) == None:
        covariates = np.zeros((S,), dtype=np.int8)
    n_cov = len(np.unique(covariates))

    # identify the sections that indicate the covariates
    cov_sections = []
    for cov in np.unique(covariates):
        _ = np.argwhere(covariates == cov).flatten()[0]
        cov_sections.append(_)

    root = zarr.open(zarr_path, mode='rb')
    mz_list = np.array(list(root.group_keys()))

    # identify the index that is closest to the given float
    assert ((v is not None) + (mz_val is not None)) > 0, 'one of mz_val or v_ind must be a value'

    if v is None:
        # find the indexes to the supplied mz values
        v = np.argmin(np.abs(mz_val - mz_list.astype(float)))
    else:
        mz_val = mz_list[v]

    fig = plt.figure(None,figsize)
    gs = plt.GridSpec(S,3)
      
    
    print(f'molecule susceptibility: {b_lambda[v].mean():.2f}')
    
    cm = plt.cm.get_cmap('bwr')
    xmin, xmax = np.percentile(x[:,:,v][mask[:,:,v]], (0.1, 99.90))
    xmax += 1.
    xnew = np.linspace(xmin, xmax, 50) 
    
    vmax_raw = np.percentile(x[:,:,v][mask[:,:,v]], 99.5)
    vmax_trans = np.percentile(x_MAIA[:,:,v][mask[:,:,v]], 99.5)
    
    for i, s in enumerate(range(S)):
        plt.subplot(gs[i,0])
        plt.hist(x[:,s,v][mask[:,s,v]], bins=30, density=True,color='gray', alpha=0.2)
        plt.hist(x_MAIA[:,s,v][mask[:,s,v]], bins=30, density=True,color='k',histtype='step', alpha=0.9)
        plt.axvline(locs[v], c="b")
        plt.plot(xnew, weights[s,v][0] * stats.norm(locs[v], scale1[v]).pdf(xnew), "b")
    
        # foreground distribution
        try:
            plt.axvline(locs[v] + delta_[s,v] + b_gamma[s] * b_lambda[v] + error[s,v], c="red")
            plt.plot(xnew, weights[s,v][1] * stats.norm(locs[v]+delta_[s,v] + b_gamma[s] * b_lambda[v]+error[s,v],
                                                          sigma_v[v] + sigma_s[s]).pdf(xnew), "red")
        except:
            print('delta')
            plt.axvline(locs[v] + delta[v] + b_gamma[s] * b_lambda[v], c="red")
            plt.plot(xnew, weights[s,v][1] * stats.norm(locs[v]+delta[v] + b_gamma[s] * b_lambda[v] + error[s,v],
                                                          sigma_v[v] + sigma_s[s]).pdf(xnew), "red")
    
        
        for c in cov_sections:
            plt.axvline(locs[v] + delta_[c,v], c="black", alpha=0.5, linestyle='--')
        plt.axvline(locs[v] + delta_[s,v], c="black")
        
        plt.xlim(xmin-2., xmax)
        plt.ylim([0,1.])
        
        
        if i == 0:
            plt.ylabel('density')
        plt.xlabel('intensity')
        
        plt.subplot(gs[i,1])
        
        img = place_image(mask_2D_list, x, v, s, np.log(epsilon))
        plt.imshow(img, cmap='Greys', interpolation='none',vmax=vmax_raw, vmin=np.log(epsilon))
        plt.axis('off')
        if i == 0:
            plt.title('Original')
            
        plt.subplot(gs[i,2])
        img = place_image(mask_2D_list, x_MAIA, v, s, np.log(epsilon))
        plt.imshow(img, cmap='Greys', interpolation='none',vmax=vmax_trans, vmin=np.log(epsilon))
        plt.axis('off')
        if i == 0:
            plt.title('Normalized')
        
    plt.colorbar()
    
    plt.suptitle(f'm/z value: {mz_val}')
    plt.tight_layout()
    plt.show()
    
            
def showMatchedImages(PATH_SAVE, mz_query, acquisitions, vmax='constant', figsize=(12,15) ):

    """
    Function to plot frequency spectrum alongside intensity spectrum for specific ranges.
    
    Args
    ----

    PATH_SAVE: str
        Path to .zarr object
    mz_query: list[float]
        List of floats corresponding to m/z from root.group_keys()
    acquisitions: List
        List of acquisition names
    vmax: str
        if 'constant' will apply the same vmax to all acquisitions per molecule. Otherwise will show a different vmax per image
    figsize: Tuple
        Indicate figure size

    """
    
    root = zarr.open(PATH_SAVE, mode='rb')
    mz_list = np.array(list(root.group_keys()))

    # find the indexes to the supplied mz values
    indexes = []
    for mz in mz_query:
        ix = np.argmin(np.abs(mz - mz_list.astype(float)))
        indexes.append(ix)
    
    
    fig = plt.figure(None,figsize,dpi=200)
    gs = plt.GridSpec(len(acquisitions), len(indexes))

    for im, mz in enumerate(mz_list[indexes]):
        images = []

        for i_s in range(len(acquisitions)):
            # add label for section name
            try:
                img = root[mz][i_s][:]
                images.append(img)
            except:
                #images.append(np.zeros(img.shape))
                images.append(np.zeros((10,10)))
        vm = np.max([np.percentile(image, 99) for image in images])
        for i_s, img in enumerate(images):
            plt.subplot(gs[i_s, im])
            if i_s == 0:
                plt.title( mz)
            if im == 0:
                plt.ylabel(acquisitions[i_s][:20], fontdict={'fontsize':4})
            try:
                if vmax =='constant':
                    plt.imshow(img, vmax=vm, interpolation='none')
                else:
                    plt.imshow(img, interpolation='none')
                plt.xticks([])
                plt.yticks([])
            except:
                plt.xticks([])
                plt.yticks([])


def place_image(masks_list, tranformed_values, v, s, epsilon):
    img = np.zeros(masks_list[s].shape).flatten()
    #img[mask_list[s].flatten()] = np.exp(tranformed_values[:np.sum(mask_list[s]),s,v]) - epsilon
    img[masks_list[s].flatten()] = tranformed_values[:np.sum(masks_list[s]),s,v]
    img[~masks_list[s].flatten().astype(bool)] = epsilon
    return img.reshape(masks_list[s].shape)

#########################################################################################################
    # added funcions aalvarezf 
#########################################################################################################

# my imports:

from pyimzml.ImzMLParser import ImzMLParser
import seaborn as sns

#########################################################################################################

def plot_ion_image_from_imzml(path_data, target_mz, tolerance=1.0, log_scale=True, cmap='magma_r', figsize=(8, 6)):
    """
    Generate and display an ion image for a given m/z from an .imzML file.
    """
    parser = ImzMLParser(path_data)

    # Determine image size
    xs = [coord[0] for coord in parser.coordinates]
    ys = [coord[1] for coord in parser.coordinates]
    width = max(xs)
    height = max(ys)

    # Create image
    image = np.zeros((height, width))

    for idx, (x, y, z) in enumerate(parser.coordinates):
        mzs, intensities = parser.getspectrum(idx)
        for mz, intensity in zip(mzs, intensities):
            if abs(mz - target_mz) <= tolerance:
                image[y - 1, x - 1] = intensity
                break

    # Plot
    plt.figure(figsize=figsize)
    data = np.log1p(image) if log_scale else image
    # plt.imshow(data, cmap=cmap, origin='lower')
    im = plt.imshow(data, cmap=cmap, origin='lower')
    plt.title(f"Ion image for m/z = {target_mz} ± {tolerance}")
    # plt.colorbar(label="Intensity (log1p)" if log_scale else "Intensity") # too big... 
    cbar = plt.colorbar(im, shrink=0.3)  # reduce la altura de la colorbar
    cbar.set_label("Intensity (log1p)" if log_scale else "Intensity", fontsize=8)  # tamaño del texto
    cbar.ax.tick_params(labelsize=8)  # tamaño de los números en la colorbar
    # plt.xlabel("")
    # plt.ylabel("")
    plt.tight_layout()
    plt.show()


# spectrum of a given pixel
def plot_mz4pix(smz, pixel): # dpi fuera ,dpi=150
    '''
        Plot m/z spectrum for a given pixel.
        choose dpi or def 150
    '''
    
    # Obtener spectro y convertir a array denso
    intensities = smz.S[pixel].toarray().flatten()
    
    # Obtener valores de m/z
    mz_vals = smz.mz_vals

    # Filtrar solo intensidades > 0?? to rethink
    mask = intensities > 0
    mz_vals_plot = mz_vals[mask]
    intensities_plot = intensities[mask]

    # output def, rollo vars no!
    # filename = f"{path_save}/{name}/mz_pixel_{pixel}.png"

    # Ploteeeea
    plt.figure(figsize=(10, 3))
    plt.plot(mz_vals_plot, intensities_plot, color='purple')
    plt.title(f"Spectrum pixel #{pixel}")
    plt.xlabel("m/z")
    plt.ylabel("intensity / relative abundance")
    plt.grid(True)
    # plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.show()


# Mean spectrum all pixels
def plot_mean_spectrum(smz, mz_range=(300, 1200)):
    """
    Plot the mean spectrum (average intensity across all pixels) from an smz object.
    """
    # promedioo 
    mean_intensity = smz.S.mean(axis=0).A1
    mz = smz.mz_vals

    plt.figure(figsize=(12, 4))
    plt.plot(mz, mean_intensity, lw=1, color='#16b87c')
    plt.xlabel("m/z")
    plt.ylabel("Mean Intensity")
    plt.title("Mean spectrum (across pixels)")
    plt.xlim(*mz_range)
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.show()

# To see coord that where measured/present in the imzML file after masking
def plot_measured_coordinates(smz, point_size=0.5, color='red', figsize=(8, 3)):
    """
    Plot the spatial coordinates of all measured spectra from an smz object.
        # this is not related to intensities, coord from smz!
    """
    coords = np.array([c[:2] for c in smz.reader.coordinates])[:smz.n_spectra]

    plt.figure(figsize=figsize)
    plt.scatter(coords[:, 0], coords[:, 1], s=point_size, color=color)
    plt.gca().invert_yaxis()
    plt.title("Measured coordinates (pixels with spectra)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Where to put the threshold in molecule matching step???? 
def plot_num_pixels_per_peak(df_list, threshold=3000, bins=100, figsize=(11, 4), alpha=0.5, palette='Set1'):
    """
    Plot histograms of 'num_pixels' for a list of DataFrames (one per df),
    with a threshold line and automatic coloring.
    """

    num_dfs = len(df_list)
    
    # Colores.. 
    if isinstance(palette, str):
        colors = sns.color_palette(palette, num_dfs)
    else:
        colors = palette[:num_dfs]

    plt.figure(figsize=figsize)

    for i, df in enumerate(df_list):
        plt.hist(df['num_pixels'], bins=bins, alpha=alpha,
                 label=f'df {i+1}', color=colors[i])

    plt.axvline(threshold, color='darkred', linestyle='--', label=f'Threshold = {threshold}')
    plt.xlabel('Number of pixels per peak')
    plt.ylabel('Number of molecules (peaks)')
    plt.title('Num pixels by peak across dfs')
    # plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # deja espacio a la derecha
    plt.show()


#########################################################################################################
    # end aaf
#########################################################################################################
    
    