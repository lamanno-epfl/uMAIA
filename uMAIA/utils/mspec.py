from __future__ import annotations  # postponed evaluation of annotations

import os
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"
os.environ["OMP_NUM_THREADS"] = "10"

import numpy as np
from scipy import sparse
import numba
from scipy.interpolate import interp1d

import logging
import math
from tqdm import tqdm as progressbar
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from typing import Tuple, Union, List, Any, Dict, Set
import more_itertools as mit

import pymzml
import pyimzml
from .ImzMLParser import ImzMLParser
import xmltodict

from threadpoolctl import threadpool_limits
# set thread limit
threadpool_limits(10)



ArrayLike = Union[List, Tuple, np.ndarray]


@numba.njit
def lipid_rule(mz: np.ndarray) -> np.ndarray:
    bool_filter = np.zeros(mz.shape[0], dtype=np.bool_)
    for i in range(len(mz)):
        integer_part = math.floor(mz[i])
        decimals = mz[i] - integer_part
        if integer_part < 700:
            if decimals >= 0.3 and decimals < 0.6:
                bool_filter[i] = True
        else:
            if decimals >= 0.4 and decimals < 0.8:
                bool_filter[i] = True
    return bool_filter



@numba.njit
def reduce_resolution_sorted(
    mz: np.ndarray, intensity: np.ndarray, resolution: float, max_intensity=True
) -> Tuple[np.ndarray, np.ndarray]:
    """Recompute a sparce representation of the spectrum a lower (fixed) resolution

    Arguments
    ---------
    mz: np.ndarray, shape=(n, )
    
    intensity: np.ndarray, shape=(n, )
    
    resolution: float
        the size of the bin used to collapse intensities

    max_intensity: bool
        if True, for the new m/z bins return the maximum intensity. Else returns additive intensity
    Returns
    -------
    """
    # First just count the unique values and store them to avoid recalc
    current_mz = -1.0
    cnt = 0

    approx_mz = np.empty(mz.shape, dtype=np.double)
    for i in range(len(mz)):
        approx_mz[i] = math.floor(mz[i] / resolution) * resolution
        if approx_mz[i] != current_mz:
            cnt += 1
            current_mz = approx_mz[i]


    new_mz = np.empty(cnt, dtype=np.double)
    new_intensity = np.empty(cnt, dtype=np.double)

    current_mz = -1.0
    rix = -1
    for i in range(len(mz)):
        if approx_mz[i] != current_mz:
            rix += 1
            new_mz[rix] = approx_mz[i]
            new_intensity[rix] = intensity[i]
            current_mz = approx_mz[i]
        else:
            # retrieve the maximum intensity value within the new bin
            if max_intensity:
                # check that the new intensity is greater than what is already there
                if intensity[i] > new_intensity[rix]:
                    new_intensity[rix] = intensity[i]

            # sum the intensity values within the new bin
            else:
                new_intensity[rix] += intensity[i]
    return new_mz, new_intensity


@numba.njit
def in1d_both_uniquesorted(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Returns the same results of np.in1d(a,b) for sorted unique-valued arrays
    It is ~20 faster than np.in1d(a,b)
    If a and b are not unique and sorted it returns nonsense and it might never stop.

    Arguments
    ---------
    a: np.ndarray
        Array to probe
    b: np.ndarray
        Array to check into

    Returns
    -------
    np.ndarray, dtype=bool, shape=(len(a),)

    Notes
    -----
    This implementation assumes strictly that:
        - a = np.sort(np.unique(a))
        - b = np.sort(np.unique(b))
    For best efficiency it should be:
        - len(a) < len(b)
    """
    len_a = len(a)
    len_b = len(b)
    i = 0
    j = 0
    res = np.zeros(len(a), dtype=np.bool_)
    while i < len_a:
        if a[i] < b[j]:
            # res[i] = False
            i += 1
        else:
            if a[i] == b[j]:
                res[i] = True
                i += 1
                j += 1  # Because of the uniqueness
            else:  # Means a[i] > b[j] This is a bit dangerous in general
                j += 1
                if j == len_b:
                    break  # res[x > i] will remain False as init
    return res

@numba.njit#(parallel=True)
def in1d_vec_nb(matrix, index_to_remove):
    #matrix and index_to_remove have to be numpy arrays
    #if index_to_remove is a list with different dtypes this 
    #function will fail

    out=np.empty(matrix.shape[0],dtype=numba.boolean)
    index_to_remove_set=set(index_to_remove)

    for i in numba.prange(matrix.shape[0]):
        if matrix[i] in index_to_remove_set:
            out[i]=False
        else:
            out[i]=True

    return out

@numba.njit#(parallel=True)
def in1d_scal_nb(matrix, index_to_remove):
    #matrix and index_to_remove have to be numpy arrays
    #if index_to_remove is a list with different dtypes this 
    #function will fail

    out=np.empty(matrix.shape[0],dtype=numba.boolean)
    for i in numba.prange(matrix.shape[0]):
        if (matrix[i] == index_to_remove):
            out[i]=False
        else:
            out[i]=True

    return out


def isin_nb(matrix_in, index_to_remove):
    #both matrix_in and index_to_remove have to be a np.ndarray
    #even if index_to_remove is actually a single number
    shape=matrix_in.shape
    if index_to_remove.shape==():
        res=in1d_scal_nb(matrix_in.reshape(-1),index_to_remove.take(0))
    else:
        res=in1d_vec_nb(matrix_in.reshape(-1),index_to_remove)

    return res.reshape(shape)


@numba.njit
def search2sorted(a: np.ndarray, v: np.ndarray, return_idx=True) -> np.ndarray:
    """Same as np.searchsorted but more efficient
    if both arrays are sorted and checks for exact identity
    If there are values in a not present in v, it never finishes.

    Arguments
    ---------
    a: np.ndarray
        Array to look into.
    
    v: np.ndarray
        Values to find in a

    Return
    ------
    ixes: np.ndarray
        Indices so that a[ixes[i]] == v[i]
    """
    MAXITER = 1_000_000_000
    len_a = len(a)
    len_v = len(v)

    
    i = 0
    j = 0
    idx_found = []
    ixes = np.empty(len_v, dtype=np.intp)
    while i < len_v:
        if v[i] == a[j]:
            ixes[i] = j
            idx_found.append(i)
            # Assumin no repeats
            i += 1
            j += 1
        else:
            j += 1
            if j == len_a:  # Finished looping throug v that no values
                if i == len_v:
                    # Everything went well
                    #print(f'ok {len_a, len_v, j, i}')
                    break
                elif return_idx==True:
                    break
                else:
                    raise StopIteration("Not found all the values of a in v")

    return ixes, i, np.array(idx_found)


@numba.njit
def fast_to_dense_fixed_mz_vals(
    fixed_mz_vals: np.ndarray, mz: np.ndarray, intensities: np.ndarray
) -> np.ndarray:
    """Returns the same results of the code below
    x = np.zeros_like(fixed_mz_vals)
    bool_f = np.in1d(fixed_mz_vals, mz)
    x[bool_f] = intensities[np.in1d(mz, fixed_mz_vals)]

    Arguments
    ---------
    fixed_mz_vals: np.ndarray, shape=(n,)
        Values corresponding to each dimension of the output
    
    mz: np.ndarray, shape=(k,)
        Values of mz indicating the species, should be a subset of fixed_mz_vals

    intensities: np.ndarray, shape=(k,)
        The corresponding intensities for each species

    Returns
    -------
    dense_intensities: np.ndarray, shape=(n,)
        The dense array of intensities with one entry for fixed_mz_vals

    Notes
    -----
    This implementation assumes strictly that:
        - mz = np.sort(np.unique(mz))
        - fixed_mz_vals = np.sort(np.unique(fixed_mz_vals))
    For best efficiency it should be:
        - len(mz) < len(fixed_mz_vals)
    """
    dense = np.zeros(len(fixed_mz_vals), dtype=np.double)  # Initialize to zero vector
    i = 0
    j = 0
    len_mz = len(mz)
    len_fixed = len(fixed_mz_vals)
    while i < len_mz:
        if mz[i] < fixed_mz_vals[j]:
            i += 1
        else:
            if mz[i] == fixed_mz_vals[j]:
                dense[j] = intensities[i]
                i += 1
                j += 1  # Because of the uniqueness
            else:  # Means a[i] > b[j] This is a bit dangerous in general
                j += 1
                if j == len_fixed:
                    break
    return dense


@numba.njit
def fast_to_dense_separated_intervals(
    separated_intervals: np.ndarray, mz: np.ndarray, intensities: np.ndarray
) -> np.ndarray:
    """Create a dense representation using the intervals
    Equivalent, but much more performant, to:
    dense = np.empty(separated_intervals.shape[0], dtype=float)
    for i in range(separated_intervals.shape[0]):
        dense[i, :] = np.sum(intensities[(mz >= separated_intervals[i,0]) & (mz < separated_intervals[i,1])])

    Arguments
    ---------
    separated_intervals: np.ndarray, shape=(n, 2)
        Values corresponding to each dimension of the output
        The intervals need to be non overlapping and sorted!
    
    mz: np.ndarray, shape=(k,)
        Values of mz indicating the species, should be a subset of fixed_mz_vals

    intensities: np.ndarray, shape=(k,)
        The corresponding intensities for each species

    Returns
    -------
    dense_intensities: np.ndarray, shape=(n,)
        The dense array of intensities with one entry for interval

    """
    dense = np.zeros(separated_intervals.shape[0], dtype=np.double)
    i = 0
    j = 0
    len_mz = len(mz)
    n_intervals = separated_intervals.shape[0]
    while i < len_mz:
        if mz[i] < separated_intervals[j][0]:  # interval is left inclusive
            i += 1
        else:
            if mz[i] < separated_intervals[j][1]:  # interval is right exclusive
                dense[j] += intensities[i]
                i += 1
            else:
                j += 1
                if j == n_intervals:
                    break
    return dense


class Spectrum:
    __slots__ = ["mz", "i"]

    def __init__(self, mz: np.ndarray, i: np.ndarray):
        self.mz = mz
        self.i = i

    def __getitem__(self, slice_like: Any) -> Spectrum:
        return Spectrum(self.mz[slice_like], self.i[slice_like])

    def __setitem__(self, slice_like: Any, value: Any) -> None:
        raise NotImplementedError("The class Spectrum is read only.")

    def copy(self) -> Spectrum:
        return Spectrum(self.mz.copy(), self.i.copy())

    def resolved(self, resolution: float) -> Spectrum:
        if resolution <= 1e-7:
            return self
        else:
            return Spectrum(*reduce_resolution_sorted(self.mz, self.i, resolution))

    def filter_mz_inplace(self, selected_mzs: np.ndarray, free_memory: bool = True) -> None:
        # assume np.all(np.in1d(selected_mzs, self.mz))
        bool_filter = in1d_both_uniquesorted(self.mz, selected_mzs)
        if free_memory:
            self.mz = self.mz[bool_filter].copy()
            self.i = self.i[bool_filter].copy()
            # Still the garbage collector needs to do the rest
        else:
            self.mz = self.mz[bool_filter]
            self.i = self.i[bool_filter]

    def mz_filtered(self, selected_mzs: np.ndarray, copy: bool = True) -> Spectrum:
        bool_filter = in1d_both_uniquesorted(self.mz, selected_mzs)
        if copy:
            return self[bool_filter].copy()
        else:
            return self[bool_filter]

    def _to_dense_fixed_mz_vals(self, fixed_mz_vals: np.ndarray) -> np.ndarray:
        return fast_to_dense_fixed_mz_vals(fixed_mz_vals, self.mz, self.i)

    def _to_dense_separated_intervals(self, separated_intervals: np.ndarray) -> np.ndarray:
        return fast_to_dense_separated_intervals(separated_intervals, self.mz, self.i)

    def to_dense(
        self, *, fixed_mz_vals: np.ndarray = None, separated_intervals: np.ndarray = None
    ) -> np.ndarray:
        if fixed_mz_vals is not None:
            return self._to_dense_fixed_mz_vals(fixed_mz_vals)
        elif separated_intervals is not None:
            return self._to_dense_separated_intervals(separated_intervals)
        else:
            raise NotImplementedError(
                "A valid method need to be passed when calling to_dense."
            )

    def sparse_ixes(self, mz_reference: np.ndarray) -> np.ndarray:
        return search2sorted(mz_reference, self.mz)


class SmzMLobjView:
    __slots__ = ["spectra", "_source_obj"]

    def __init__(self, spectra: np.ndarray, source_obj: Any) -> None:
        self.spectra = spectra
        self._source_obj = source_obj

    def __getitem__(self, slice_like: Any) -> SmzMLobjView:
        return SmzMLobjView(self.spectra[slice_like], self.source_obj)

    def __getattr__(self, attr: str) -> Any:
        # warnings.warn("Accessing source object attribute")
        return getattr(self._source_obj, attr)

    def __len__(self) -> int:
        return len(self.spectra)


class SmzMLobj:
    """Spatial mzML file Object

        Parameters
        ----------
        filepath: str
            Path to a .mzML file (it can be gzipped)
        
        annotation: str or dict, default = None
            Filepath to an annotation file or a dictionary containing annotation.
            For now, only .UDP is supported.
        
        mz_resolution: float, default = 1e-7
            The resolution to load and process the data.
            Recomended small upon loading, as it can be reduced
            later, during analysis.
        
        autoload: bool, default = False
            Whether to load the data upon initialization of the object.
            If False, a connection to the file is established and part of the
            data will be read when it is required by other actions or when
            desired using the method `load`
        
        **pymzml_kwargs: Dict[Any, Any],

        Attributes
        ----------
        mz_resolution: float
            The current resolution of the data loaded in the file.
            It can be changed using the `fix_resolution` methods
        
        resolutionx: int
            The actual resolution over the x axis in microns.

        resolutiony: int
            The actual resolution over the y axis in microns.

        massmin: float
            The minimum value of the spectrum recorded.

        massmax: float
            The maximum value of the spectrum recorded.

        annot_dict: dictionary
            Contains all the annotation passed at initialization
            or extracted from the .UDP file specified.

        S: sparse.csc_matrix, shape=(n_spectra, n_mz_vals)
            A sparse matrix containing a dense representation of the
            spectra intensities each value S[i, j] corresponds
            to the spectrum `i` and the species with mass `mz_vals[j]`

        A: np.ndarray, shape=(spectra, species)
            An array containing a dense representation of the
            spectra intensities each value A[i, j] corresponds
            to the spectrum `i` and the species with mass `mz_vals[j]`
            WARNING: It is recommended to check `mz_vals` before loading
            to avoid the allocation of a huge array in memory!

        img_shape: tuple
            The shape (rows, columns) of the recorded area.

        img: np.ndarray, shape=(rows, cols, species)
            The same as A but formated as a 3d np.ndarray

        mz_vals: int
            The m/z values approximated to the currently selected resolution

        n_mz_vals: int:
            return len(self.mz_vals)

        n_spectra: int
            return self.__len__()

        shape: tuple
            For convenience the pair self.n_spectra, self.n_mz_vals

        Notes
        -----
        The keyword arguments for the pymzml are:
        MS_precisions: float, default= None
        obo_version: Any, default = None
        build_index_from_scratch: bool, default = False
        skip_chromatogram: bool, default = True
    """

    DTYPE_I = np.float64

    def __init__(
        self,
        filepath: str,
        annotation: Union[Dict[str, Any], str] = None,
        mz_resolution: float = 1e-7,
        autoload: bool = False,
        selected_pixels = None,
        mz_shift: dict = None,
        **pymzml_kwargs: Dict[Any, Any],
    ) -> None:

        

        self.filepath = filepath
        self._mz_resolution = mz_resolution
        self.selected_pixels = selected_pixels
        self.kwargs = pymzml_kwargs
        
        
        self.scandirection: int
        self.resolutionx: int
        self.resolutiony: int
        self.maxx: int
        self.maxy: int
        self.massmin: float
        self.massmax: float


        # load readers
        if annotation.lower().endswith('imzml'):
            self.reader = ImzMLParser(annotation,
                ibd_file=self.filepath,
               parse_lib='lxml')
            self.num_spectra = len(self.reader.coordinates)
        elif annotation.lower().endswith('udp'):
            self.reader = pymzml.run.Reader(self.filepath, **self.kwargs)
            self.num_spectra = self.reader.get_spectrum_count()

        if annotation is None:
            self.annot_filepath = None
            self.annot_dict = {}
            self.expected_pixels = self.reader.get_spectrum_count()
        elif os.path.splitext(annotation)[-1].lower() == ".imzml":
            self.annot_filepath = annotation
            self.annot_dict = self.reader.imzmldict
            try:
                self.maxx = self.annot_dict['max count of pixels x']
                self.maxy = self.annot_dict['max count of pixels y']
                self.resolutionx = self.annot_dict['pixel size x']
                self.resolutiony = self.annot_dict['pixel size y']
            except:
                logging.warning("The annotation dictionary has faulty keys. To check metadata, use attribute annot_dict of the smz object")
            

            self.expected_pixels = len(self.reader.coordinates)
            n_extra_spectra = len(self.reader.coordinates) - self.expected_pixels

            if n_extra_spectra > 0:
                logging.warning(
                    f"The mzML file containts more than the exprected {self.expected_pixels} spectra. "
                    + f"The tailing {n_extra_spectra} spectra will be ignored."
                )

        elif os.path.splitext(annotation)[-1].lower() == ".udp":
            

            self.annot_filepath = annotation
            self.annot_dict = xmltodict.parse(open(self.annot_filepath).read())


            # Automatically set those scan variables
            scan_dict_types = {
                "ScanDirection": int,
                "ResolutionX": int,
                "ResolutionY": int,
                "MaxX": int,
                "MaxY": int,
                "MassMin": float,
                "MassMax": float,
            }
            for key, dtype in scan_dict_types.items():
                setattr(self, key.lower(), dtype(self.annot_dict["UDPPrj"]["Scan"][key]))
            self.expected_pixels = self.maxx * self.maxy
            n_extra_spectra = self.reader.get_spectrum_count() - self.expected_pixels

            # print(f'Expected pixels={self.expected_pixels}')
            # print(f'Spectrumcount from reader={self.reader.get_spectrum_count()}')
            if n_extra_spectra > 0:
                logging.warning(
                    f"The mzML file containts more than the exprected {self.expected_pixels} spectra. "
                    + f"The tailing {n_extra_spectra} spectra will be ignored."
                )
        else:
            raise IOError(
                f"File of type {os.path.splitext(annotation)[-1].lower()} is not supported as annotation file."
            )


        

        if self.selected_pixels:
            self.expected_pixels = self.selected_pixels[1] - self.selected_pixels[0]
        if self.expected_pixels > self.num_spectra:
            self.expected_pixels = self.num_spectra
        self.spectra = np.empty(self.expected_pixels, dtype=np.object_)
        self.spectra_original = []
        self._I: np.ndarray = None  # shape = (pixels, self._mz_vals_at_res)
        self._small_access_cnt = 0
        self._mz_vals_at_res: Dict[float, np.ndarray] = {}
        
        self.mz_shift = mz_shift
        if self.mz_shift is None:
            self.mz_shift = np.zeros(self.expected_pixels)
            

    def __len__(self) -> int:
        return len(self.spectra)

    def get_mz_vals(
        self, mz_resolution: float = None, force_recompute: bool = False
    ) -> List[float]:
        """Get all the m/z values as the set union of the ones in the spectra

        Arguments
        ---------
        mz_resolution: float, default = None
            The resolution to use to get the the mz values.
            If None (reccomended) it extracts the mz_vals
            at resolution self.mz_resolution

        force_recompute: bool, default = False
            Whether to force recomputation set union
            of mz_vals that was previously found

        Return
        ------
        mz_vals: np.ndarray
            If mz_resolution is None or == self.mz_resolution
            the values are both returned and stored
            Otherwise mz_vals is only returned and not stored.
        """
        mz_vals: Set
        if mz_resolution is None:
            mz_resolution = self.mz_resolution
        if mz_resolution not in self._mz_vals_at_res:
            # This case is triggered when the data at a ceratain resolution was never loaded
            # This case get_mz_vals will not store anything in the object
            mz_vals = set()
            for spectrum in progressbar(
                self.spectra,
                desc=f"Evaluating the m/z values at resolution {self.mz_resolution}",
            ):
                mz_vals.update(spectrum.resolved(mz_resolution).mz)
            return mz_vals
        elif force_recompute or self._mz_vals_at_res[mz_resolution] is None:
            mz_vals = set()
            for i_spectrum, spectrum in progressbar(
                enumerate(self.spectra),
                desc=f"Loading the m/z values at resolution {self.mz_resolution}",
            ):
                try:
                    mz_vals.update(spectrum.mz)
                except:
                    # print(f'I broke {i_spectrum}')
                    continue
            # Store and return
            self._mz_vals_at_res[mz_resolution] = np.array(sorted(mz_vals))
            return mz_vals
        else:
            # The values where already computed and stored
            return self._mz_vals_at_res[mz_resolution]

    def recompute_mz_vals(self, mz_resolution: float = None) -> List[float]:
        return self.get_mz_vals(mz_resolution, force_recompute=True)

    def change_resolution(self, mz_resolution: float, inplace: bool = True) -> SmzMLobj:
        """Change the resolution of the spectra. I cannot be reverted.

        Arguments
        ---------
        mz_resolution: float
            The effective resolution of the spectrum to be obtained.
            `mz_values` that are closer than `mz_resolution` will be collapsed.
            The change will affect all the following steps
        
        inplace: bool = True;
            whether to perform the change of resulution inplace
            or return a different object at `mz_resolution` (NOT IMPLEMENTED YET)

        Returns
        -------
        self if inplace=True, otherwise a new `SmzMLobj`

        """
        previous_bigger_res = max(list(self._mz_vals_at_res.keys()), default=0)
        if mz_resolution < previous_bigger_res:
            raise ValueError(
                "You cannot chose a resolution finer than the one previously set. Reload the object to achieve it it."
            )
        elif mz_resolution < 4 * previous_bigger_res:
            logging.warning(
                "Careful! This new resolution is not much coarser than the one previously set. Binning might result influenced."
            )
        else:
            pass
        if inplace:
            self._mz_resolution = mz_resolution
            mz_vals: Set = set()
            for i in progressbar(range(len(self.spectra)), desc="Changing spectra resolution"):
                tmp = self.spectra[i].resolved(mz_resolution)
                mz_vals.update(tmp.mz)
                self.spectra[i] = tmp
            self._mz_vals_at_res[self._mz_resolution] = np.array(sorted(mz_vals))
            return self
        else:
            raise NotImplementedError(
                "A fixed resolution view is not inplemented yet. Use inplace=True"
            )

    def filter_spectra(self, mz_bool_condition: np.ndarray, inplace: bool = True) -> None:
        """The general methods that discards a subset of m/z peaks

        Arguments
        ---------
        mz_bool_condition: np.ndarray, shape=(n_mz_vals,)
            A boolean array that is True in correspondece to a mz_vals to keep.

        inplace: bool = True
            Whether to perform the filterin inplace or
            return a filtered version of the object (NOT IMPLEMENTED YET)
        """
        if inplace:
            selected_mzs = self._mz_vals_at_res[self.mz_resolution][mz_bool_condition]
            for i in progressbar(
                range(len(self.spectra)),
                desc=f"Filtering {mz_bool_condition.sum()} peaks out of {len(self._mz_vals_at_res[self.mz_resolution])}",
            ):
                self.spectra[i].filter_mz_inplace(selected_mzs)
            self._mz_vals_at_res[self.mz_resolution] = selected_mzs
        else:
            raise NotImplementedError(
                "A deep filtered view is not inplemented yet. Use inplace=True"
            )

    def filter_lipids(self) -> None:
        lipid_bool_condition = self._lipid_condition(self.mz_vals)
        self.filter_spectra(lipid_bool_condition, inplace=True)

    @staticmethod
    def _lipid_condition(mz_vals: np.ndarray) -> np.ndarray:
        return lipid_rule(mz_vals)

    def _densify(self) -> None:
        self._I = np.empty(self.shape, dtype=self.DTYPE_I)
        for i in range(self.n_spectra):
            self._I[i, :] = self.spectra[i].to_dense(fixed_mz_vals=self.mz_vals)

    def _generate_sparse_intensities(self) -> sparse.lil_matrix:
        S = sparse.lil_matrix(self.shape, dtype=np.double, copy=False)
        for i, spectrum in progressbar(enumerate(self.spectra), total=len(self.spectra_original)):

            ix_sort = np.argsort(spectrum.mz)
            mz_sort = spectrum.mz[ix_sort]
            i_sort = spectrum.i[ix_sort]


            #indices = np.where(np.in1d(self.mz_vals, mz_sort))[0]
            try:
                indices = np.where(~isin_nb(self.mz_vals,mz_sort))[0]
                #indices = np.where(in1d_both_uniquesorted(self.mz_vals, mz_sort))[0]
                S[i, indices] = i_sort
            except Exception as e:
                print(e)
                print(S.shape)
                print(len(ix_sort), len(i_sort))
                print(len(indices))
                print(indices)
                print(isin_nb(self.mz_vals,mz_sort))
                # indices = np.where(np.in1d(self.mz_vals, mz_sort))[0]
                # S[i,indices] = i_sort
            

        return S.tocsc()

    def recompute_A(self) -> None:
        self._densify()

    def recompute_S(self) -> None:
        self._I = self._generate_sparse_intensities()

    @property
    def mz_resolution(self) -> float:
        return self._mz_resolution

    @property
    def S(self) -> sparse.lil_matrix:
        if self._I is None:
            self._I = self._generate_sparse_intensities()
            return self._I
        elif sparse.issparse(self._I):
            return self._I
        else:
            logging.warning(
                "A dense matrix was already computed. Returning a sparsified dense matrix!"
            )
            return sparse.lil_matrix(self._I)

    @property
    def A(self) -> np.ndarray:
        if self._I is None:
            self._densify()
            return self._I
        elif sparse.issparse(self._I):
            return self._I.toarray()
        else:
            return self._I

    @property
    def img_shape(self) -> Tuple[int, int]:
        return (self.maxy, self.maxx)

    @property
    def img(self) -> np.ndarray:
        return self.A.reshape((self.n_spectra, self.maxy, self.maxx))

    @property
    def mz_vals(self) -> np.ndarray:
        return self.get_mz_vals(self._mz_resolution, force_recompute=False)

    @property
    def n_mz_vals(self) -> int:
        return len(self.mz_vals)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.n_spectra, self.n_mz_vals

    @property
    def n_spectra(self) -> int:
        return self.__len__()

    def _load(self, ixs2load: np.ndarray, show_progress: bool = False) -> None:
        set_ixs2load = set(
            ixs2load
        )  # ixs2load should be unique, I just do this for fast lookup later.
        n_toload = len(set_ixs2load)
        cnt = 0
        if show_progress:
            with progressbar(
                total=n_toload, desc=f"Loading Spectra at resolution {self.mz_resolution}"
            ) as pbar:
                if type(self.reader) != ImzMLParser:
                    for i, spectrum in enumerate(self.reader):
                        if i in set_ixs2load:
                            self.spectra[cnt] = Spectrum(spectrum.mz - (self.mz_shift[cnt] * spectrum.mz), spectrum.i).resolved(
                                self.mz_resolution
                            )
                            self.spectra_original.append(spectrum)
                            cnt += 1
                            pbar.update(1)
                        if cnt == n_toload:
                            break
                else:
                    for i, (x,y,z) in enumerate(self.reader.coordinates):
                        if i in set_ixs2load:
                            mz, intensity = self.reader.getspectrum(i)
                            self.spectra[cnt] = Spectrum(mz - (self.mz_shift[cnt] * mz), intensity).resolved(
                                self.mz_resolution
                            )
                            cnt += 1
                            pbar.update(1)
                            if cnt == n_toload:
                                break
        else:
            if type(self.reader) != ImzMLParser:
                for i, spectrum in enumerate(self.reader):
                    if i in set_ixs2load:
                        self.spectra[i] = Spectrum(spectrum.mz - (self.mz_shift[cnt] * spectrum.mz), spectrum.i).resolved(
                            self.mz_resolution
                        )
                        cnt += 1
                    if cnt == n_toload:
                        break
            else:
                for i, (x,y,z) in enumerate(self.reader.coordinates):
                    if i in set_ixs2load:
                        mz, intensity = self.reader.getspectrum(i)
                        self.spectra[cnt] = Spectrum(mz - (self.mz_shift[cnt] * mz), intensity).resolved(
                            self.mz_resolution
                        )
                        cnt += 1
                        if cnt == n_toload:
                            break

    def load(self, load_unique_mz: bool = False) -> None:
        """Load in memory the contents of the mzML file
        
        Arguments
        ---------
        load_unique_mz: bool, default = False
            Whether to load an array of all the m/z values encountered.
            If False, the attribute `mz_vals` will be loaded the first
            time it is required for any computation.
        """
        
        if self.selected_pixels:
            ixs2load = np.arange(len(self)) + self.selected_pixels[0]
        else:
            ixs2load = np.arange(len(self))
        # print(ixs2load.shape)
        self._load(ixs2load, show_progress=True)
        # Put a non placeolder, to remember that the data has been loaded at this resolution
        self._mz_vals_at_res[self.mz_resolution] = None

        if load_unique_mz:
            self.get_mz_vals()

    def __getitem__(self, slice_like: Any) -> SmzMLobjView:
        is_none = self.spectra[slice_like] == None  # In this case it should be like this
        # Check if any of the spectra has not been loaded
        if np.any(is_none):
            ixs2load = np.arange(len(self.spectra))[slice_like][is_none]
            if len(ixs2load) < 20:
                self._small_access_cnt += 1
                if self._small_access_cnt >= 10:
                    logging.warning(
                        "Accessing small number of new spectra often is not adviced. "
                        + "Use load, to load them all once using `load()`"
                    )
            self._load(ixs2load)
        return SmzMLobjView(self.spectra[slice_like], self)

    def _extract(
        self, sel_spectra: ArrayLike, sel_peaks: np.ndarray, copy: bool = False
    ) -> List[List[np.ndarray]]:
        """Internal function to extract a subset of peaks for a set of spectra
        """
        out_list = []
        for i in sel_spectra:
            spectrum = self.spectra[i].mz_filtered(sel_peaks, copy=copy)
            out_list.append([spectrum.i, spectrum.mz])
        return out_list

    def iplot_spectra(
        self,
        sel_spectra: ArrayLike,
        sel_peaks: Union[str, np.ndarray],
        cmap: Any = plt.cm.rainbow,
        show_legend: bool = True,
        fig: Any = None,
        ax: Any = None,
        alpha: float = 0.7,
        **plotkwargs: Dict[str, Any],
    ) -> None:
        """Plot a set interactive spectra showing only some peaks
        This is make for interactive visualization as zoom will be required to see the peaks.
        
        DO NOT USE THIS FUNCTION FOR STATIC PLOTS!

        Parameters
        ----------
        sel_spectra: ArrayLike
            Selected spectra. A list or array containing integers indicating the index of the pixels/spectra. e.g. [100, 120, 1002, ...]
            If a list of 2-element-list or 2d array it will interpreted as (row, colums pairs)

        sel_peaks: np.ndarray,
            Selected Peaks. 
            
            If an array. It will have to contain the values of the m/z of the peaks to show.
            It is normally obtained by subsetting `self.mz_vals`, for example it could be the result of something like:
            smz.mz_vals[(smz.mz_vals > 856) & (smz.mz_vals < 876)]

            If a string. It can contain a query, using the placeholder name `mz` to specify the m/z
            The expression should correspond to valid python code. For examples for the examples above it should be:
            "(mz > 856) & (mz < 876)"
            WARNING: this is a dangerous hack as it uses `eval`!

        cmap: Any = plt.cm.rainbow,
            If a it will be used differentiate the different

        show_legend: bool = True,
            Whether to show the legend

        fig: Any = None,
            If a figure it is specify the plot will be shown it its correspondence

        ax: Any = None
            If an axes is passed a new figure will be created

        alpha: float = 0.7
            The transparency controlling parameter

        Return
        ------
        ax:
            It produces a matplotlib plot and teturn the axes
        """
        kwargs: Dict[str, Any] = {"ec": None}
        kwargs.update(plotkwargs)

        if isinstance(sel_peaks, str):
            sel_peaks = sel_peaks.replace("mz", "self.mz_vals")
            sel_peaks = self.mz_vals[eval(sel_peaks)]

        exsp = self._extract(sel_spectra, sel_peaks)

        if ax is None:
            if fig is None:
                fig = plt.figure(None, (10, 4))
            elif isinstance(fig, tuple):
                fig = plt.figure(None, fig)
            ax = fig.add_subplot(111)

        ax.set_xlim(np.min(sel_peaks), np.max(sel_peaks))

        for i, (intensities, mz_values) in enumerate(exsp):
            if isinstance(cmap, mpl_colors.Colormap):
                fc = cmap(i / len(exsp))
            else:
                fc = cmap
            b = ax.bar(
                mz_values,
                intensities,
                width=self.mz_resolution,
                fc=fc,
                alpha=alpha,
                label=sel_spectra[i],
                **kwargs,
            )
        ax.set_ylim(0,)

        if show_legend:
            fontsize = max(4, min(16, 6 * 15 / len(sel_spectra)))
            ax.legend(fontsize=fontsize)

        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")

        return ax

    def plot_spectra(
        self,
        sel_spectra: ArrayLike,
        sel_peaks: Union[str, np.ndarray],
        bar_width_in_scale: bool = False,
        lw: float = 2,
        cmap: Any = plt.cm.rainbow,
        show_legend: bool = True,
        fig: Any = None,
        ax: Any = None,
        alpha: float = 0.7,
        **plotkwargs: Dict[str, Any],
    ) -> None:
        """Plot a set non-interactive spectra showing only some peaks.
        This is make static visualization or to visualize spectra when one does not care of peaks overlapping.

        Parameters
        ----------
        sel_spectra: ArrayLike
            Selected spectra. A list or array containing integers indicating the index of the pixels/spectra. e.g. [100, 120, 1002, ...]
            If a list of 2-element-list or 2d array it will interpreted as (row, colums pairs)

        sel_peaks: np.ndarray,
            Selected Peaks. 
            
            If an array. It will have to contain the values of the m/z of the peaks to show.
            It is normally obtained by subsetting `self.mz_vals`, for example it could be the result of something like:
            smz.mz_vals[(smz.mz_vals > 856) & (smz.mz_vals < 876)]

            If a string. It can contain a query, using the placeholder name `mz` to specify the m/z
            The expression should correspond to valid python code. For examples for the examples above it should be:
            "(mz > 856) & (mz < 876)"
            WARNING: this is a dangerous hack as it uses `eval`!

        bar_width_in_scale: bool = True
            If True the width of the bar will be, otherwise it will be set just to the passed value `lw`
        
        lw: float = 2
            the width of the line

        cmap: Any = plt.cm.rainbow,
            If a it will be used differentiate the different

        show_legend: bool = True,
            Whether to show the legend

        fig: Any = None,
            If a figure it is specify the plot will be shown it its correspondence

        ax: Any = None
            If an axes is passed a new figure will be created

        alpha: float = 0.7
            The transparency controlling parameter

        Return
        ------
        ax:
            It produces a matplotlib plot and return the axes
        """
        kwargs: Dict[str, Any] = {"fc": "none", "lw": lw}
        kwargs.update(plotkwargs)

        if isinstance(sel_peaks, str):
            sel_peaks = sel_peaks.replace("mz", "self.mz_vals")
            sel_peaks = self.mz_vals[eval(sel_peaks)]

        exsp = self._extract(sel_spectra, sel_peaks)

        if ax is None:
            if fig is None:
                fig = plt.figure(None, (10, 4))
            elif isinstance(fig, tuple):
                fig = plt.figure(None, fig)
            ax = fig.add_subplot(111)

        ax.set_xlim(np.min(sel_peaks), np.max(sel_peaks))
        if bar_width_in_scale:
            _lw = linewidth_from_data_units(self.mz_resolution, axis=ax, reference="x")
            kwargs["lw"] = _lw
        for i, (intensities, mz_values) in enumerate(exsp):
            if isinstance(cmap, mpl_colors.Colormap):
                kwargs["ec"] = cmap(i / len(exsp))
            else:
                kwargs["ec"] = cmap
            b = ax.vlines(
                mz_values, 0, intensities, alpha=alpha, label=sel_spectra[i], **kwargs,
            )
        ax.set_ylim(0,)

        if show_legend:
            fontsize = max(4, min(16, 6 * 15 / len(sel_spectra)))
            ax.legend(fontsize=fontsize)

        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")

        return ax

    def peak_windows_trace(
        self,
        sel_peaks: Union[ArrayLike, List[ArrayLike], np.ndarray],
        sel_spectra: ArrayLike = None,
    ) -> ArrayLike:
        """It produces a vector containing, for every spectrum, the center of mass computed of the selected peaks 
        Most often sel_peaks contains a window of sel_peaks

        Arguments
        ---------
        sel_peaks: [1d-array or list] or [2d-array, list-of-lists]
            A set of Selected Peaks or multiple sets Selected Peaks.
            If one ste of Selected Peaks:
                It contains the values of the m/z of the peaks to consider.
                It is normally obtained by subsetting `self.mz_vals`, for example it could be the result of something like:
                smz.mz_vals[(smz.mz_vals > 856) & (smz.mz_vals < 876)]
            If multiple set of Selected Peaks:
                A list of lists/arrays (or a 2d array) containing the peaks as explained above
        
        sel_spectra: ArrayLike = None
            If None, all the spectra will be considered.

        Returns
        -------
        CoM_list: List[np.ndarray]
            A list containing for each set of Selected Peaks a trace.
            If sel_peaks is a 1d array or simple list it just retruns a 1d array
        """
        if sel_spectra is None:
            sel_spectra = np.arange(self.n_spectra)
        sel_peaks = _sel_peaks_parse(sel_peaks)
        CoM_list = []
        for sel in sel_peaks:
            extracted = self._extract(sel_spectra, sel)
            CoM = np.zeros(len(extracted))
            for i, (intensities, mzs) in enumerate(extracted):
                if len(intensities) == 0:
                    CoM[i] = np.nan
                else:
                    # Center of mass calculation
                    CoM[i] = np.sum(intensities * mzs) / np.sum(intensities)
            CoM_list.append(CoM)
        if len(CoM_list) == 1:
            return CoM_list[0]
        else:
            return CoM_list

    def plot_peak_window_trace(
        self,
        sel_peaks: Union[ArrayLike, List[ArrayLike]],
        lw: float = 1,
        fig: Any = None,
        ax: Any = None,
        show_legend: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        """It produces a plot that shows, for every spectrum, the center of mass computed
        accross a set of selected peaks (usually a window of peaks)

        Arguments
        ---------
        sel_peaks: [1d-array or list] or [2d-array, list-of-lists]
            A set of Selected Peaks or multiple sets Selected Peaks.
            If one ste of Selected Peaks:
                It contains the values of the m/z of the peaks to consider.
                It is normally obtained by subsetting `self.mz_vals`, for example it could be the result of something like:
                smz.mz_vals[(smz.mz_vals > 856) & (smz.mz_vals < 876)]
            If multiple set of Selected Peaks:
                A list of lists/arrays (or a 2d array) containing the peaks as explained above

        lw: float = 1:
            The width of the plotted line

         fig: Any = None,
            If a figure it is specify the plot will be shown it its correspondence

        ax: Any = None
            If an axes is passed a new figure will be created

        show_legend: bool = False
            Whether to show a figure legend

        kwargs**
            The function accepts any keywords argument accepted by
            the plt.plot function, to adjust the haestetics.
            With exception of color and label,

        Returns
        -------
        ax:
            It produces a matplotlib plot and return the axes obejct
        """

        # Preprocess the input so for it to be flexible
        sel_peaks = _sel_peaks_parse(sel_peaks)
        CoM_list = self.peak_windows_trace(sel_peaks)
        if len(sel_peaks) == 1:
            CoM_list = [CoM_list]

        if ax is None:
            if fig is None:
                fig = plt.figure(None, (9.5, 2))
            elif isinstance(fig, tuple):
                fig = plt.figure(None, fig)
            ax = fig.add_subplot(111)

        for i, CoM in enumerate(CoM_list):
            color = f"C{i}"
            ax.plot(CoM, lw=lw, c=color, label=f"Window {i+1}", **kwargs)
            ax.axhline(np.min(CoM[~np.isnan(CoM)]), c=color, ls="--", lw=lw)
            ax.axhline(np.max(CoM[~np.isnan(CoM)]), c=color, ls="--", lw=lw)
        ax.set_ylabel("m/z")
        ax.set_xlabel("Pixels (scan ord.)")

        if show_legend:
            fontsize = max(4, min(16, 6 * 15 / len(CoM_list)))
            ax.legend(fontsize=fontsize)

        return ax

    def visualize_bin_range_deviation_intensity(
        self,
        bin_range: tuple,
        percentiles: tuple=(30, 50, 70),
        plot_graph: bool=False,
        plot_style: str='com',
        alpha: float=0.7,
	    save_image= False,
        ax=None):

        """Function takes in a range of mz values as a tuple and outputs a scatter plot with axis deviation from averaged CoM values and intensity per pixel. Returns also indexes for pixels which contain a signal of interest as well as the indexes of the mz bins which satisfy the specified range. The percentiles will return the mean intensity and standard deviation of the signal for each percentile bin.
        
        Arguments
        ---------
        bin_range: tuple(float, float)
            The m/z range that one wants to visualize across pixels  
        
        percentiles: tuple(int)
            The percentile values according to intensity for which one wants to calculate the average intensity and standard deviation of the CoM per pixel to the averaged CoM

        plot_graph: bool
            True if one wants to plot

        plot_style: str
            'com' if one wants to retrieve the center of mass in the plot for each pixel
            'total' if one wants to plot all peaks for a pixel in the graph
        
        alpha: float
            Transparency value for points on scatterplot

        save_image
            String for the path if one wants to save. Else leave as False
        
        ax: matplotlib.plt.axis object
            The axis for which one wants to place the plot. Else leave as is
        

        Returns
        -------
        INDEX: np.ndarray
            The indexes of the columns of S for mz bins which contain a signal
        
        INDEX_PIXEL: np.ndarray
            The indexes of the rows of S for which pixels contain at least one signal from the specified bin range

        CoM_std: np.ndarray
            The standard deviations of the mz CoM for each specified percentile range
        
        percentile_mean_intensity: np.ndarray
            The average of the intensity values for the specified percentile intensity ranges
        
        pixel_bin_presence: np.ndarray
            For each pixel, a counter for how many times a signal in the bin range was picked up.
            
        
        
        """

        INDEX = (self.mz_vals > bin_range[0]) & (self.mz_vals < bin_range[1]) 
        INDEX = INDEX.flatten()
        IMG_SUM = self.S[:,INDEX].toarray().sum(axis=1)
        INDEX = np.argwhere(INDEX == True).flatten()
        colors_time = plt.cm.cool(np.linspace(0,1,self.img_shape[0] * self.img_shape[1]))

        # select pixels that express some value for the m/z range
        IDX_PIXEL = np.argwhere((IMG_SUM > 0 )).flatten()

        # calculate the best approximation for the true m/z of species in provided bin
        true_mz = self.S[:,INDEX.flatten()].toarray().astype(bool).sum(axis=0)
        true_mz = np.sum(true_mz / true_mz.sum() * self.mz_vals[INDEX.flatten()])   

        S_pixels = self.S[IDX_PIXEL][:,INDEX.squeeze()]
        selected_colors = colors_time[IDX_PIXEL] 

        # for each pixel, how many peaks were present in the bin?
        pixel_bin_presence = np.sum(S_pixels > 0, axis=1).flatten()

        if plot_graph:
            if ax == None:
                fig, ax = plt.subplots()

            for i, row in enumerate(S_pixels):
                
                non_zero_idx = np.argwhere(row > 0)[:,1]

                if plot_style == 'com':
                    
                    # calculate CoM for pixel
                    weight = row[0,non_zero_idx] / row[0,non_zero_idx].sum()
                    CoM = weight * self.mz_vals[INDEX.squeeze()[non_zero_idx]]
                    # calculate difference between CoM and 'true' value
                    deviation = CoM# - true_mz
                    intensity = row[0,non_zero_idx].mean()
                    ax.scatter(intensity, deviation,
                     color=selected_colors[i],
                      marker='x', alpha=alpha)

                elif plot_style == 'total':

                    for value in non_zero_idx:

                        mz = self.mz_vals[INDEX.squeeze()[value]]
                        deviation = mz# - true_mz
                        intensity = row[0,value]
                        ax.scatter(intensity, deviation,
                         color=selected_colors[i], 
                         marker='x', alpha=alpha)


                ax.set_xlabel('Intensity')
                ax.set_ylabel('Devation from average m/z value')

            if save_image:
                fig.savefig(save_image)

        # calculate the standard deviation among percentiles
        percentiles_ = [np.percentile(IMG_SUM[IMG_SUM > 0], x) for x in percentiles]
        percentiles_ = np.insert(percentiles_, 0,0)
        CoM_std = []
        percentile_mean_intensity = []
        

        for i in range(len(percentiles_)):
            if i == len(percentiles_) - 1:
                idx_pixel = np.argwhere(IMG_SUM > percentiles_[i]).flatten()
            else:
                idx_pixel = np.argwhere((IMG_SUM > percentiles_[i] ) & (IMG_SUM < percentiles_[i + 1])).flatten()
            S_pixels = self.S[idx_pixel][:,INDEX.squeeze()]

            # calculate CoM for each pixel
            CoMs = []
            for i, row in enumerate(S_pixels):
                non_zero_idx = np.argwhere(row > 0)[:,1]
                # calculate CoM for pixel
                weight = row[0,non_zero_idx] / row[0,non_zero_idx].sum()
                CoM = weight * self.mz_vals[INDEX.squeeze()[non_zero_idx]]
                CoMs.append(CoM)
            CoMs = np.array(CoMs)
            # take array of pixel CoMs and calculate standard deviation
            CoM_std_ = CoMs.std()
            CoM_std.append(CoM_std_)

            # we want an idea of the intensity for each percentile
            # take all values that are non zero in the selection and calculate average
            percentile_mean_intensity_ = np.mean(S_pixels[S_pixels > 0])
            percentile_mean_intensity.append(percentile_mean_intensity_)

        return INDEX, IDX_PIXEL, CoM_std, percentile_mean_intensity, pixel_bin_presence

    def visualize_bin_range_image(
        self,
        bin_range: List[Tuple],
        range_type: str='bins',
        plot_graph: bool=True,
        vmin:int=2,
        vmax:int=98,
        normalize:bool=True,
        save_image=False,
        ax=None):
        
        """Function to visualize the cumulative signal on image from a given mz range

        Arguments
        ---------
        bin_range: List[Tuple]
            The m/z ranges that one wants to visualize across pixels

        range_type: str
            'bins' where the input to bin_range is a List[Tuple]
            'explicit' where the input to bin_range is a List of m/z values one wants to include. If the value does not exist in self.mz_vals then it is ignored
        
        plot_graph: bool
            True if one wishes to visualize the image. Otherwise, closes the image immediately so that they can be saved

        vmin: int
            Percentile of the image sum to trim by
        
        vmax: int
            Percentile of the image sum to put ceiling to

        save_image:
            String for the path if one wants to save. Else leave as False
        
        ax: matplotlib.plt.axis object
            The axis for which one wants to place the plot. Else leave as is
        """
        



        if range_type == 'bins':
            # initialize zeros array
            total_boolean_array = np.zeros(self.n_mz_vals, dtype=bool)

            # iterate over passed ranges
            for i,(low_thresh, high_thresh) in enumerate(bin_range):
                total_boolean_array += (self.mz_vals > low_thresh) & (self.mz_vals < high_thresh)
                
            # retrieve indexes where there are signals
            boolean_array = np.argwhere(total_boolean_array == True).flatten()
        
        elif range_type == 'explicit':
            boolean_array = [np.argwhere(x == self.mz_vals).flatten()[0] for x in bin_range]



        matrix = self.S[:,boolean_array] # consider only specified indexes in S
        
        # normalize so that the sum of each row is the same
        if normalize:
            matrix = matrix / np.sum(matrix, axis=1)
        
        # calculate image sum across ranges
        IMG_SUM_ = matrix.sum(axis=1).reshape(self.img_shape)
        
        # visualize image

        if plot_graph:
            if ax == None:
                fig, ax = plt.subplots()
            ax.imshow(IMG_SUM_, cmap='inferno',
            vmin=np.percentile(IMG_SUM_, vmin), 
            vmax=np.percentile(IMG_SUM_, vmax)
            )
            ax.axis('off')

            if range_type == 'explicit':
                    ax.set_title(f'min: {np.min(bin_range)}, max: {np.max(bin_range)}')   
            else:
                ax.set_title(bin_range)

            # save image
            if save_image:
                fig.savefig(save_image)

        return IMG_SUM_

    def retrieve_closest_peak(
        self,
        mz_value: float):
        
        """Function to determine closest peak to passed mz_value
        
        Arguments
        ---------
        mz_value: float
            The m/z value to query

        Returns
        -------
        mz_close: float
            The closest m/z value for which there was a registered signal

        above, idx_above: Tuple[float, int]
            Tuple for which first value is the closest value above the mz value and the second value is the index of self.mz_vals for which the value occurs
        
        below, idx_below: Tuple[float, int]
            Tuple for which first value is the closest value below the mz value and the second value is the index of self.mz_vals for which the value occurs

        """

        # define indexes that are below and above specified value
        idx_above = np.argwhere(mz_value <= self.mz_vals).flatten().min()
        idx_below = np.argwhere(mz_value >= self.mz_vals).flatten().max()
        # determine the actual value
        above, below = self.mz_vals[idx_above], self.mz_vals[idx_below]
        # find which is closest
        idx_closeness = np.argmin(
            (np.abs(mz_value - above), np.abs(mz_value - below))
            )
        mz_close = [above, below][idx_closeness]

        return mz_close, (above, idx_above), (below, idx_below) 
            


def _sel_peaks_parse(sel_peaks: Any) -> ArrayLike:
    """Correctly format sel_peaks so to accept both 1d/2d arrays/lists/list-of-lists
    """
    if isinstance(sel_peaks, np.ndarray):
        if len(sel_peaks.shape) > 1:
            if sel_peaks.shape[0] == 2 and sel_peaks.shape[1] != 2:
                sel_peaks = list(sel_peaks)
            elif sel_peaks.shape[0] != 2 and sel_peaks.shape[1] == 2:
                sel_peaks = list(sel_peaks.T)
            else:
                raise ValueError(f"Input of shape {sel_peaks.shape} is ambiguous")
        else:
            sel_peaks = [sel_peaks]
    elif isinstance(sel_peaks, list):
        if isinstance(sel_peaks[0], list) or isinstance(sel_peaks[0], np.ndarray):
            sel_peaks = sel_peaks
        else:
            sel_peaks = [sel_peaks]
    else:
        raise ValueError(f"Entry of format {type(sel_peaks)} is not a valid entry.")
    return sel_peaks


def linewidth_from_data_units(linewidth: float, axis: plt.Axes, reference: str = "x") -> float:
    """Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    if reference == "x":
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == "y":
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    else:
        raise ValueError(f"`{reference}`` is not a valid axis")
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return linewidth * (length / value_range)
