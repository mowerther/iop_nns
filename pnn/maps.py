import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py

l1_path = os.path.join(r"C:\SwitchDrive\Papers\iop_ml\prisma_imagery", 
                       r"PRS_L1_STD_OFFL_20220309101419_20220309101424_0001.he5")
l2_dir = r"C:\SwitchDrive\Papers\iop_ml\prisma_imagery\prisma_map_outputs"
l2_prod = "PRISMA_2022_03_09_10_14_19_L2W-bnn_dc-prisma_gen_l2_iops.nc"
l2w_dir = r"C:\SwitchDrive\Papers\iop_ml\prisma_imagery"
l2w_prod = "PRISMA_2022_03_09_10_14_19_L2W.nc"
output_dir = r"C:\github_repos\eawag\PRISMA"

# hdf5 for the L1 product
with h5py.File(l1_path, 'r') as f:
    vnir_cube = f['/HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube'][:]
    
    # some RGB bands, can also pick other bands
    red = vnir_cube[:, 32, :]    # Band 33
    green = vnir_cube[:, 45, :]  # Band 46
    blue = vnir_cube[:, 61, :]   # Band 62
    
    # create normalize RGB composite
    rgb = np.dstack((red, green, blue))
    
    def normalize_band(band):
        p2, p98 = np.percentile(band, (2, 98))
        return np.clip((band - p2) / (p98 - p2), 0, 1)

    rgb_normalized = np.dstack([normalize_band(rgb[:,:,i]) for i in range(3)])
    rgb_normalized = np.rot90(rgb_normalized, k=-1)

# load L2 product - L2W or so
l2_file = os.path.join(l2_dir, l2_prod)
l2w_file = os.path.join(l2w_dir, l2w_prod)
ds = xr.open_dataset(l2_file)
l2w_ds = xr.open_dataset(l2w_file)

# water mask and masked array
water_mask = (l2w_ds.l2_flags == 0) | (l2w_ds.l2_flags == 8)
masked_aph = np.ma.masked_array(ds['aph_443'].values, mask=~water_mask)
masked_aph_std = np.ma.masked_array(ds['aph_443_std_pct'].values, mask=~water_mask)

# percentiles for both variables
p2_aph, p98_aph = np.percentile(masked_aph.compressed(), (2, 98))
p2_std, p98_std = np.percentile(masked_aph_std.compressed(), (2, 98))

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

viridis = plt.cm.magma
cividis = plt.cm.cividis
viridis.set_bad('none')
cividis.set_bad('none')

# left subplot - aph443
ax1.imshow(rgb_normalized)
im1 = ax1.imshow(masked_aph, vmin=p2_aph, vmax=p98_aph, cmap=viridis, alpha=0.8)
cbar1 = plt.colorbar(im1, ax=ax1, format='%.3f', label=r'a$_{ph}(443)$ [m$^{-1}$]', extend='both')
ticks1 = np.linspace(p2_aph, p98_aph, 6)
cbar1.set_ticks(ticks1)
ax1.set_title(r'PRISMA L2 a$_{ph}(443)$')
ax1.axis('off')

# right subplot - aph443 std percentage
ax2.imshow(rgb_normalized)
im2 = ax2.imshow(masked_aph_std, vmin=p2_std, vmax=p98_std, cmap=cividis, alpha=0.8)
cbar2 = plt.colorbar(im2, ax=ax2, format='%.1f', label=r'a$_{ph}(443)$ std [%]', extend='both')
ticks2 = np.linspace(p2_std, p98_std, 6)
cbar2.set_ticks(ticks2)
ax2.set_title(r'PRISMA L2 a$_{ph}(443)$ unc.')
ax2.axis('off')

plt.tight_layout()
 
plt.savefig(os.path.join(output_dir, 'prisma_l2_aph443_and_std_with_rgb_background.png'), 
            dpi=300, 
            bbox_inches='tight')
plt.close()

# close datasets
ds.close()
l2w_ds.close()