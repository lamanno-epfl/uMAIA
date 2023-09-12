import os
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
from matplotlib.ticker import *
import more_itertools as mit

# multithreading
# from threadpoolctl import threadpool_limits

# sklearn and scipy imports
from scipy.ndimage import gaussian_filter1d

# set thread limit
# threadpool_limits(4)


class PeakFinder:

    """The PeakFinder class contains methods that will fit a Gaussian Mixture model to an smzObj that it is passed by using the distribution of m/z values provided by all pixels in the image space. This is uncommon in mass spectrometry analysis, where one will typically be interested in the intensity values of m/z spectra
    
    Args
    ----
    smz: SmzMLobj
        the SmzMLobj containing the loaded data and specified resolution parameter
    mz_range: tuple
        a tuple containing the range of m/z values one is interested in fitting peaks to. default is 0 to 15000
    threshold_count: int
        the threshold below which values are removed if the given m/z bin does not contain a sufficient number of detections. default 2
    num_components: int
        the number of components of gaussians that should be fit the the range of interest
    df_gmm: pd.DataFrame
        dataframe containing the mz ratio in one column and the class number in the second column. Used for plotting functions
    """

    def __init__(self, S, mz_vals, mz_range:tuple=(0,15000), threshold_count:int=50, 
                smoothing=2, means_init=None, mz_resolution=0.0001):
        self.S = S
        self.mz_vals = np.array(mz_vals)
        self.mz_range = np.array(mz_range).astype(np.float64)
        self.ix = (self.mz_vals >= self.mz_range[0]) & (self.mz_vals <= self.mz_range[1])
        self.mz_vals_within_range = self.mz_vals[self.ix]
        self.threshold_count = threshold_count
        self.S_compressed = None
        self.idx_filtered = None
        self.means_init = means_init
        self.ranges = np.array([[np.nan, np.nan]])
        self.smoothing = smoothing
        self.mz_resolution = mz_resolution


    def process(self):
        """The peak finder can be applied to an smzObj with the aim of determining where peaks exist in the m/z spectrum.
        The function will create a data array in one dimensions which will be used for the peak finding algorithm """

        # check that the queries mz_range is valid
        if np.sum(self.mz_vals[(self.mz_vals >= self.mz_range[0]) & (self.mz_vals <= self.mz_range[1])]) == 0:
            print('invalid mz_range')
            return False

        # if means_init is supplied, restrict the values so that they fall in the mz range
        if self.means_init is not None:
            means_init_idx = (self.means_init >= self.mz_range[0]) & (self.means_init <= self.mz_range[1])
            self.means_init = self.means_init[means_init_idx]
            self.num_components = len(self.means_init)
        
            
        # binarize the S matrix and sum across the rows
        all_mz = np.concatenate(np.array(self.S[:,self.ix].todense()).astype(bool).astype(int) * self.mz_vals_within_range)
        self.S_compressed, self.data_mz = np.histogram(all_mz[np.nonzero(all_mz)],
                                  bins=np.arange(self.mz_range[0],
                                                 self.mz_range[1],
                                                 self.mz_resolution * 7))

        
        self.mz_range = np.argwhere(self.ix == True).flatten()

        try:
            self.mz_range = (self.mz_range.min(), self.mz_range.max())
        except Exception as e:
            self.data_1d = np.array([])
            return False

        self.S_smooth = gaussian_filter1d(self.S_compressed, sigma=self.smoothing)

        self.idx_filtered = self.S_compressed > self.threshold_count

        # extract frequency and value of each mz
        self.data_mz = self.data_mz[:-1]

        # initialize 1d array
        self.data_1d = np.array([])

        # iterate over mz values and append to another data frame the same value x times according to freq
        for i, d in enumerate(self.data_mz):
            freq = self.S_smooth[i]
            rep = np.repeat(d, freq) # duplicate mz
            self.data_1d = np.append(self.data_1d, rep)
            
        if len(self.data_1d) <= 1: # operation has no data for points with frequency greater than threshold
            return False
        else:
            return True # operations succeeded

    def fit(self):
        """Function will fit the desired gaussian mixture model to the specified data range in the mass spec output.
        Args
        ----
        threshold_likelihood: float
            When provided, will initiate the component number optimization. The float value provided here
            corresponds to the threshold cut off of the average loglikelihood during the first round of 
            component number optimization.
        
        Returns
        -------
        df_gmm: DataFrame
            data frame containing the class number assigned to each m/z bin
        """

        # determine the optimal n_components, rewrites self.num_components
        thresholds = np.arange(self.threshold_count , self.S_smooth.max(), 1)[::-1]

        self.sequences = []

        #initialize the component list
        self.seeds = [np.argwhere(self.S_smooth > self.S_smooth.max() - 1).flatten().tolist()]
        
        # we want to identify the bin centers
        for i, threshold in enumerate(thresholds[1:]):
            data = np.argwhere(self.S_smooth > threshold).flatten()
            # retrieve indexes of consecutive components
            seq_continguous = [list(group) for group in mit.consecutive_groups(data)]
            # for each sequence, merge with parent index
            self.sequences.append(seq_continguous)

        for i_s, sequence in enumerate(self.sequences):
            # append if seed is in contig that hasn't been seen before
            for contig in sequence:
                if len(set(contig).intersection(set(np.concatenate(self.seeds)))) == 0:
                    self.seeds.append([contig[0]])


        # we can sort the seeds list
        self.seeds = np.sort(np.concatenate(self.seeds))

        ranges_final = []
        # for each seed, select the index where S_smooth goes below the threshold, or increases
        for seed in self.seeds:
            index_left = seed - 1
            index_right = seed + 1
            

            # stay in the while loop as long as (1) the value of S_smooth on the left is greater than the value to the right (2) is value of S_smooth is still above the threshold count (3) the index is not out of bounds
            while (self.S_smooth[index_left - 1] < self.S_smooth[index_left]) and (self.S_smooth[index_left] > self.threshold_count) and (index_left > 0):
                index_left -= 1

            # check that the bound is within the range
            if index_right < len(self.S_smooth) -1:
                while (self.S_smooth[index_right + 1] < self.S_smooth[index_right]) and (self.S_smooth[index_right] > self.threshold_count) and (len(self.S_smooth) - 2 > index_right):
                    index_right += 1
            else:
                index_right = len(self.S_smooth) - 2

            ranges_final.append([index_left, index_right])
            
        ranges_checked = []
        # check that range does not contain more than one seed
        for range_ in ranges_final:
            if len(set(self.seeds).intersection(np.arange(*range_))) == 1:
                ranges_checked.append(np.clip(range_,0,np.inf).astype(int))

        self.ranges_index = ranges_checked
        self.ranges = []

        for p in ranges_checked:
            if len(p) == 0:
                continue
            elif len(p) == 1 :
                self.ranges.append(np.array([self.data_mz[p], self.data_mz[p]]).flatten())
            else:
                self.ranges.append([self.data_mz[p].min(), self.data_mz[p].max()])


        self.ranges = np.array(self.ranges)