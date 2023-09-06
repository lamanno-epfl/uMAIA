import matplotlib.pyplot as plt
import numpy as np
from ..utils.tools import extract_image_coordinates
from ..peak_finding import PeakFinder
import pandas as pd


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


def image_mz(smz, df:pd.DataFrame, mz_list:list, figsize:tuple=(15,15), ylim=None, cmap='inferno', img_shape=None, limit=0.01, normalize='tic', clip=99, coordinates=None):
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
            img = extract_image_coordinates(coordinates, img_shape, img)
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