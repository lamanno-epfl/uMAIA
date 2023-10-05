import argparse
from ._run import run
from utils.mspec import SmzMLobj
import os
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = 'Run peakfinder')
    parser.add_argument('-i', '--input', metavar='input', required=True, type=str, help='Path to acquisition file')
    parser.add_argument('-o', '--output', metavar='output', required=True, type=str, help='Path to output file')
    parser.add_argument('-i_m', '--input_metadata', metavar='input_metadata', required=True, type=str, help='Path to acquisition metadata file')
    parser.add_argument('-s', '--spectrum', metavar='spectrum', required=True,  nargs='+', type=int, help='Tuple indicating extents of spectrum to scan')
    parser.add_argument('-r', '--resolution', metavar='resolution', default=10e-5, required=False, type=float, help='Resolution to load object')
    parser.add_argument('-sm', '--smoothing', metavar='smoothing', default=3, required=False, type=float, help='Factor to smooth spectrum')
    parser.add_argument('-t', '--threshold', metavar='threshold', default=15, required=False, type=int, help='Threshold count')
    parser.add_argument('-a_interval', '--a_interval', metavar='a_interval', default=1.0, required=False, type=float, help='Approximated interval for mz partitioning')
    parser.add_argument('-p', '--parallelize', metavar='parallelize', required=False, default='true', type=str, help='whether to run proceses in parallel')
    args = parser.parse_args()
    
    
    print('Loading object')

    # parse paralleluze input
    if args.parallelize == 'true':
        parallelize = True
    else:
        parallelize = False

    smz = SmzMLobj(args.input, args.input_metadata,
                mz_resolution=args.resolution)#, selected_pixels=(0,367*230))
    smz.load(load_unique_mz=True)
    #smz.spectra = smz.spectra[np.argwhere(smz.spectra != None).flatten()]

    smz.S
    print('Running peakcaller')
    run(directory_path=args.output,
    smz=smz, 
    spectrum_range=tuple(args.spectrum), threshold_count=args.threshold, approximate_interval=args.a_interval,
    smoothing=args.smoothing,
    parallelize=parallelize)

    print("Completed")