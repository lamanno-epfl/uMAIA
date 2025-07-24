
import anndata
import tqdm
import numpy as np



df_ranges

# check whether num_inside or num_outside column is missing
if ('conc_outside' in df_ranges.columns) and ('conc_inside' in df_ranges.columns):
    continue


tmp_h5ad = uMAIA.ut.tools.load_ann(annobject)#anndata.read_h5ad(annobject) 
threshold = 0.005
mz_values = tmp_h5ad.var.values.flatten()
idx_lipid = []
mz_values_ref = []
for i, mz in enumerate(df_ranges.mz_estimated.values):
    # extract the image
    argmin = np.argmin(np.abs(mz_values - mz))
    if np.abs(mz_values - mz)[argmin] <= threshold:
        idx_lipid.append(argmin)
        mz_values_ref.append(i)
    else:
        print('f')

idx_lipid = np.array(idx_lipid).astype(int)
tmp_h5ad = tmp_h5ad.X[:,idx_lipid]

# calculate the number of values inside the mask
num_inside = []
num_outside = []

conc_inside = []
conc_outside = []

# calculate the sum of intensities inside and outside the mask

for i in range(len(df_ranges)):
    # if the value is low, there is a lot more expression outside the fish
    num_inside_ = len(np.nonzero(tmp_h5ad[:,i][mask.flatten().astype(bool)])[0])#.sum()/ tmp_h5ad[:,i][~mask.flatten().astype(bool)].sum()
    num_outside_ = len(np.nonzero(tmp_h5ad[:,i][~mask.flatten().astype(bool)])[0])
    
    conc_inside_ = tmp_h5ad[:,i][mask.flatten().astype(bool)].sum()
    conc_outside_ = tmp_h5ad[:,i][~mask.flatten().astype(bool)].sum()
    
    # append to dataframe
    num_inside.append(num_inside_)
    num_outside.append(num_outside_)
    conc_inside.append(conc_inside_)
    conc_outside.append(conc_outside_)
    
# append column to dataframe
df_ranges['num_inside'] = num_inside
df_ranges['num_outside'] = num_outside
df_ranges['conc_inside'] = conc_inside
df_ranges['conc_outside'] = conc_outside

# overwrite the dataframe
df_ranges.to_csv(ranges_files[ix_select])

# filter stuff!
df_ranges = df_ranges[df_ranges.conc_outside < df_ranges.conc_inside * 1.3]  # by concentration of pixels
