from ._moleculematch import MoleculeMatcher
import os

os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

import pandas as pd
import numpy as np
import networkx as nx
import tqdm
# multithreading
from threadpoolctl import threadpool_limits
threadpool_limits(4)



def match(MAX_DIST, df_list, parameters=None):

    df_concat = np.concatenate([x.mz_estimated.values for x in df_list])
    range_ = np.floor(df_concat).min(), np.ceil(df_concat).max()
    ranges = np.column_stack([np.arange(*range_), np.arange(*range_) + 1.])


    df_matched = pd.DataFrame(columns = ['molecule_ID','min','max','mz_estimated', 'section_ix', 'concentration'])
    MOL_ID = 0
    ROW_ID = 0
    
    for i, RANGE in tqdm.tqdm(enumerate(ranges), total=len(ranges)):
        mz, NUM_S, NUM_PERMS = set_up(df_list, RANGE, parameters)

        if len(np.concatenate(mz)) <= 1 or len(np.nonzero([len(x) for x in mz])[0]) <= 1:
            continue
            
        
        set_list, permutation = retrieve_setlist(mz=mz, NUM_S=NUM_S, NUM_PERMS=NUM_PERMS, MAX_DIST=MAX_DIST)
        if len(set_list) == 0:
            for section_ix in permutation:
                df_tmp = df_list[section_ix]
                df_tmp = df_tmp[(df_tmp.mz_estimated >= RANGE[0]) & (df_tmp.mz_estimated <=RANGE[1])]
                for name, selected_row in df_tmp.iterrows():
                    MOL_ID += 1
                    row = [MOL_ID, selected_row['min'], selected_row['max'], selected_row.mz_estimated, section_ix, selected_row.concentration]
                    df_matched.loc[ROW_ID] = row
                    ROW_ID += 1
                
        else:
        # iterate over groups
            for i_s, set_ in enumerate(set_list):
                MOL_ID += 1

                for ii, detection in enumerate(set_):

                    df_tmp = df_list[permutation][detection[0]]
                    df_tmp = df_tmp[(df_tmp.mz_estimated >= RANGE[0]) & (df_tmp.mz_estimated <=RANGE[1])]
                    selected_row = df_tmp.iloc[detection[1]]

                    r = selected_row['min'], selected_row['max']

                    section_ix = np.arange(len(permutation))[permutation[detection[0]]]


                    row = [MOL_ID, selected_row['min'], selected_row['max'], selected_row.mz_estimated, section_ix, selected_row.concentration]
                    df_matched.loc[ROW_ID] = row
                    ROW_ID += 1

    df_matched = df_matched.merge(df_matched.groupby('molecule_ID')['mz_estimated'].mean(), on='molecule_ID', suffixes=('', '_total'))
    return df_matched


def set_up(df_list, RANGE, parameters):
    mz = []
    for df in df_list:
        df_tmp = df[(df.mz_estimated >= RANGE[0]) & (df.mz_estimated <=RANGE[1])]
        mz.append(df_tmp.mz_estimated.values)
        
    mz = np.array(mz, dtype=object)
    if parameters:
        NUM_S = parameters['NUM_S']
        NUM_PERMS = parameters['NUM_PERMS']
    else:
        NUM_S = len(mz)
        NUM_PERMS = NUM_S * 2
    return mz, NUM_S, NUM_PERMS

def retrieve_setlist(mz, NUM_S, NUM_PERMS, MAX_DIST):
    MM = MoleculeMatcher(mz=mz, NUM_SKIP=int(np.ceil(NUM_S / 2)), NUM_PERMS=NUM_PERMS, MAX_DIST=MAX_DIST)
    try: 
        permutation, output, selected_edges, obj_val, edge_cost, edges_unique = MM.assess_permutations()
    except:
        return [], np.arange(NUM_S)
    set_list = []

    for set_ in integrate_output(mz, selected_edges, permutation):
        sublist = []
        for ix in set_:
            node = get_tuple_from_ix(ix,mz, permutation)
            sublist.append(node)

        set_list.append(sublist)
        
    return set_list, permutation


def filter_matches(df_match, num_match=None):
    if num_match is None:
        num_match = len(np.unique(df_match.section_ix))
    df_filter = df_match[df_match.groupby('molecule_ID')['molecule_ID'].transform('size') >= num_match]
    return df_filter


# write function to retrieve index in image file
def retrieve_index(image_file, mz):
    image_mz = image_file.var.values.flatten()
    ix = np.argmin(np.abs(image_mz - mz))

    return ix

def retrieve_matched(selected_mz:float, df_match:pd.DataFrame, image_list:list, section_ix=None):
    # extract from df_filter the values from mz_estimated_total close to this value
    ix = np.argmin(np.abs(df_match.mz_estimated_total - selected_mz))
    df_sub = df_match[df_match.mz_estimated_total == df_match.iloc[ix].mz_estimated_total]
    
    if section_ix is None:
        images = {}
        for name, row in df_sub.iterrows():
            section_mz = row.mz_estimated
            section_ix = int(row.section_ix)
            ix = retrieve_index(image_list[section_ix], section_mz)
            img = image_list[section_ix].X[:,ix]

            images[section_ix] = [section_mz, img]
            
    else:

        row = df_sub[df_sub.section_ix == section_ix]
        section_mz = row.mz_estimated.values
        if len(section_mz) == 0:
            images = np.nan
        else:
            ix = retrieve_index(image_list[section_ix], section_mz)

            # check if image_list is the hdf5 file itself or a list of files
            if isinstance(image_list, list):
                images = section_mz, image_list[section_ix].X[:,ix]
            else:
                images = section_mz, image_list.X[:,ix]
            
    return images



def get_index_from_stack(section_position: tuple, mz, permutation):
    """
    retrieve index of edge given tuple of (section, ix_detection)
    """
    cumsum = np.insert(np.cumsum([len(x) for x in mz[permutation]]),0,0)
    section_ix = cumsum[section_position[0]]
    # need to add the mz detection ix
    ix = section_ix + section_position[1]
    return ix
    
def get_tuple_from_ix(ix, mz, permutation):
    cumsum = np.insert(np.cumsum([len(x) for x in mz[permutation]]),0,0)
    s_ix = np.max(np.argwhere(cumsum <= ix).flatten())
    d_ix = ix - cumsum[s_ix]
    return (s_ix, d_ix)


def integrate_output(mz, selected_edges, permutation):   
    """
    retrieve the sets of connected components given the list of lists mz detections, selected edges and permutation
    """
    ### EVALUATE ###
    # create a matrix that is nodes vs nodes, with 1 in place where an edge was placed in our output
    output_matrix = np.zeros((len(np.concatenate(mz)), len(np.concatenate(mz)))) # initialize matrix
    for edge in selected_edges:
        if 's' in edge or 'e' in edge:
            continue
        ix0 = get_index_from_stack(edge[0], mz, permutation)
        ix1 = get_index_from_stack(edge[1], mz, permutation)
        output_matrix[ix0, ix1] = 1
          
    connected_comps = list(nx.algorithms.components.connected_components(nx.Graph(output_matrix)))
    
    return connected_comps