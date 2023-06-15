import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

from tqdm import tqdm
import os
import pickle

from pathlib import Path
BASE = Path(__file__).parent.parent

print(BASE)

DATADIR = BASE / 'data' / 'st001-20230525-test'
assert DATADIR.exists()

MYDATADIR = BASE / 'my_data' / 'hm3'
MYDATADIR.mkdir(parents=True, exist_ok=True)

FIGDIR = BASE / 'my_figure' / 'hm3'
FIGDIR.mkdir(parents=True, exist_ok=True)

SEX_WORKSPACE = BASE / 'sex' / 'hm3'
SEX_WORKSPACE.mkdir(parents=True, exist_ok=True)

SOLVE_FIELD_PATH = Path('/usr/local/astrometry/bin/solve-field')
assert SOLVE_FIELD_PATH.exists()

CACHEDIR = BASE / 'cache'
CACHEDIR.mkdir(parents=True, exist_ok=True)


def func_show(img, pmin=5, pmax=95, title='', saveto=''):
    plt.figure()
    vmin = np.percentile(img, pmin)
    vmax = np.percentile(img, pmax)
    plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    if saveto:
        plt.savefig(saveto, dpi=300)
    plt.close()


def get_fpath_lsts():
    bias_fpath_lst = list(DATADIR.glob('st001-bias-20230525-*.fit'))
    flat_fpath_lst = list(DATADIR.glob('st001-flat-20230525-*.fit'))
    img_fpath_lst = list(DATADIR.glob('st001-M101-20230525-*.fit'))

    def gen_shape_mask(fits_lst, target_shape=(6388, 9576), tag=None):
        mask = []
        for idx, fits_path in tqdm(
                enumerate(fits_lst),
                total=len(fits_lst),
                desc=tag,
        ):
            try:
                data = fits.getdata(fits_path)
                if data.shape != target_shape:
                    mask.append(False)
                else:
                    mask.append(True)
            except:
                mask.append(False)
        return mask


    bias_mask = gen_shape_mask(bias_fpath_lst, target_shape=(6388, 9576), tag='bias')
    bias_fpath_lst = [bias_fpath_lst[i] for i in range(len(bias_fpath_lst)) if bias_mask[i]]
    print(f'bias: {len(bias_fpath_lst):10d}')

    flat_mask = gen_shape_mask(flat_fpath_lst, target_shape=(6388, 9576), tag='flat')
    flat_fpath_lst = [flat_fpath_lst[i] for i in range(len(flat_fpath_lst)) if flat_mask[i]]
    print(f'flat: {len(flat_fpath_lst):10d}')

    img_mask = gen_shape_mask(img_fpath_lst, target_shape=(6388, 9576), tag='image')
    img_fpath_lst = [img_fpath_lst[i] for i in range(len(img_fpath_lst)) if img_mask[i]]
    print(f'img:  {len(img_fpath_lst) :10d}')

    return bias_fpath_lst, flat_fpath_lst, img_fpath_lst


def stack_bias(bias_fpath_lst):
    assert len(bias_fpath_lst) > 0, 'No bias fits file found!'
    bias_lst = []
    N_bias_ctr = 0
    for bias_fpath in tqdm(
            bias_fpath_lst,
            total=len(bias_fpath_lst),
            desc='reading bias',
    ):
        try:
            bias_lst.append(fits.getdata(bias_fpath))
            N_bias_ctr += 1
        except:
            continue

    # collect to get an 3d array
    bias_cube = np.array(bias_lst)
    del bias_lst
    print(f'bias shape: {bias_cube.shape}')

    bias = np.nanmedian(bias_cube, axis=0)  
    # bias_var = np.nanvar(bias_cube, axis=0) # read noise
    del bias_cube
    print(f'stacked bias shape: {bias.shape}')
    return bias


def process_flat(flat_fpath_lst, bias):
    flat_lst = []
    N_flat_ctr = 0

    for flat_fpath in tqdm(
            flat_fpath_lst,
            total=len(flat_fpath_lst),
            desc='reading flat',
    ):
        try:
            flat_lst.append(fits.getdata(flat_fpath) - bias)
            N_flat_ctr += 1
        except:
            continue

    # collect to get an 3d array
    flat_cube = np.array(flat_lst)
    del flat_lst
    print(f'flat shape: {flat_cube.shape}')

    flat_median_arr = np.median(flat_cube, axis=(1,2))
    for i in range(flat_cube.shape[0]):
        flat_cube[i] = flat_cube[i] / flat_median_arr[i]

    flat = np.median(flat_cube, axis=0)

    return flat


def gen_sex_config(img_name, header):
    with open(SEX_WORKSPACE / 'default.sex', 'r') as sex_in:
        sex_lines = sex_in.readlines()
    with open(SEX_WORKSPACE / f'{img_name}.sex', 'w') as sex_out:
        for line in sex_lines:
            if line.startswith('CATALOG_NAME'):
                jd_obs = header['JD']
                t_exp = header['EXPTIME']
                line = line[:17] + '{:<14s}'.format(f'{img_name}_{jd_obs:020.10f}_{t_exp:010.5f}.cat') + line[31:]
            if line.startswith('PARAMETERS_NAME'):
                line = line[:17] + '{:<14s}'.format(f'{img_name}.param') + line[31:]
            if line.startswith('PHOT_APERTURES'):
                # aperture diameter from 20pix to 60pix step 5 pix
                # arg_str = ','.join([str(d) for d in range(20, 65, 5)])
                # line = line[:17] + '{:<14s}'.format(arg_str) + line[31:]
                line = line[:17] + '{:<14s}'.format('40') + line[31:]
            if line.startswith('PIXEL_SCALE'):
                line = line[:17] + '{:<14s}'.format('0') + line[31:]
            sex_out.write(line)


    with open(SEX_WORKSPACE / 'default.param', 'r') as sex_in:
        sex_lines = sex_in.readlines()
    with open(SEX_WORKSPACE / f'{img_name}.param', 'w') as sex_out:
        for line in sex_lines:
            if line[1:26].strip() in (
                'NUMBER', 
                'FLUX_ISO', 
                'FLUXERR_ISO', 
                'MAG_ISO', 
                'MAGERR_ISO', 
                'FLUX_APER', 
                'FLUXERR_APER', 
                'MAG_APER', 
                'MAGERR_APER', 
                'FLUX_BEST', 
                'FLUXERR_BEST', 
                'MAG_BEST', 
                'MAGERR_BEST', 
                'BACKGROUND', 
                'THRESHOLD', 
                'X_IMAGE', 
                'Y_IMAGE', 
                'ALPHA_J2000', 
                'DELTA_J2000', 
            ):
                line = ' ' + line[1:25] + '#' + line[26:]   # un-comment
            sex_out.write(line)


def main():
    TO_GEN = False
    if (CACHEDIR / 'bias_path.txt').exists():
        with open(CACHEDIR / 'bias_path.txt', 'r') as fid:
            lines = fid.readlines()
        bias_fpath_lst = [Path(line.strip()) for line in lines]
    else:
        TO_GEN = True
    if (CACHEDIR / 'flat_path.txt').exists():
        with open(CACHEDIR / 'flat_path.txt', 'r') as fid:
            lines = fid.readlines()
        flat_fpath_lst = [Path(line.strip()) for line in lines]
    else:
        TO_GEN = True
    if (CACHEDIR / 'img_path.txt').exists():
        with open(CACHEDIR / 'img_path.txt', 'r') as fid:
            lines = fid.readlines()
        img_fpath_lst = [Path(line.strip()) for line in lines if len(line.strip()) > 0]
    else:
        TO_GEN = True
    if TO_GEN:
        bias_fpath_lst, flat_fpath_lst, img_fpath_lst = get_fpath_lsts()
        with open(CACHEDIR / 'bias_path.txt', 'w') as fid:
            for fpath in bias_fpath_lst:
                fid.write(fpath.as_posix() + '\n')
        with open(CACHEDIR / 'flat_path.txt', 'w') as fid:
            for fpath in flat_fpath_lst:
                fid.write(fpath.as_posix() + '\n')
        with open(CACHEDIR / 'img_path.txt', 'w') as fid:
            for fpath in img_fpath_lst:
                fid.write(fpath.as_posix() + '\n')


    if (CACHEDIR / 'bias.cache.npy').exists():
        bias = np.load(CACHEDIR / 'bias.cache.npy')
    else:
        bias = stack_bias(bias_fpath_lst)
        np.save(CACHEDIR / 'bias.cache', bias)
    func_show(bias, title='bias', saveto=FIGDIR / 'bias.png')

    if (CACHEDIR / 'flat.cache.npy').exists():
        flat = np.load(CACHEDIR / 'flat.cache.npy')
    else:
        flat = process_flat(flat_fpath_lst, bias)
        np.save(CACHEDIR / 'flat.cache.npy', flat)
    func_show(flat, title='flat', saveto=FIGDIR / 'flat.png')

    # process image
    N_imgs = len(img_fpath_lst)
    for idx, img_fpath in enumerate(img_fpath_lst):
        print('')
        # if len(list(SEX_WORKSPACE.glob(f'{img_fpath.stem}*.cat'))):
        #     print(idx+1, N_imgs, 'found catalog for', img_fpath, 'skipped')
        #     continue
        if not 'st001-M101-20230525-0001' in img_fpath.stem:
            continue
        else:
            print(idx+1, N_imgs, 'doing', img_fpath)
        img = (fits.getdata(img_fpath) - bias) / flat
        header = fits.getheader(img_fpath)
        func_show(img, title='img '+img_fpath.stem, saveto=FIGDIR / ('img_'+img_fpath.stem + '.png'))
        
        sigma_clip = SigmaClip(sigma=3.0)
        bkg_estimator = MedianBackground()
        bkg = Background2D(img, (200, 200), filter_size=(3, 3),
                        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

        func_show(bkg.background, title='bkg '+img_fpath.stem, saveto=FIGDIR / ('bkg_'+img_fpath.stem + '.png'))

        img_clean = img - bkg.background
        func_show(img_clean, title='img_clean '+img_fpath.stem, saveto=FIGDIR / ('img_clean_'+img_fpath.stem + '.png'))

        fits.writeto(MYDATADIR / ('clean_' + img_fpath.name), data=img_clean, header=header, overwrite=True)
        print('saved to', MYDATADIR / ('clean_' + img_fpath.name))

        img_wcs_basename = 'wcs_' + img_fpath.stem
        os.system(f'{SOLVE_FIELD_PATH} --cpulimit 60 --resort --scale-units arcsecperpix --scale-low 0.2 --scale-high 0.5 --no-plots --overwrit --dir {SEX_WORKSPACE} --out {img_wcs_basename} {img_fpath} -O')

        img_wcs_fpath = SEX_WORKSPACE / (img_wcs_basename + '.new')
        gen_sex_config(img_fpath.stem, header)
        os.system(f'cd {SEX_WORKSPACE} && sex {img_wcs_fpath} -c {img_fpath.stem}.sex')

        # (MYDATADIR / ('clean_' + img_fpath.name)).unlink()



if __name__ == '__main__':
    main()