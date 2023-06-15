import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import convolve2d
from scipy.optimize import leastsq
from astropy.time import Time

from tqdm import tqdm
import pickle

from pathlib import Path
BASE = Path(__file__).resolve().parent.parent


DATADIR = BASE / 'data'
assert DATADIR.exists(), f'{DATADIR} not exists!'

CACHEDIR = BASE / 'cache'
CACHEDIR.mkdir(parents=True, exist_ok=True)


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


# def cal_gain_single(flat, bias_var):
#     flat_mean = np.nanmean(flat, axis=(0, 1))
#     read_noise2_adu = np.nanmean(bias_var, axis=(0, 1))
#     flat_var = np.nanvar(flat, axis=(0, 1)) - read_noise2_adu

#     gain_cal = flat_mean / flat_var
#     read_noise = np.sqrt(read_noise2_adu) * gain_cal
#     return gain_cal, read_noise, flat_var, flat_mean
    

def cal_gain(flat_cube, bias_var, save_fig=None):
    '''
    return gain, chi2dof
    '''
    flat_mean_arr = np.nanmean(flat_cube, axis=(1, 2))
    read_noise2 = np.nanmean(bias_var, axis=(0, 1))
    flat_var_arr = np.nanvar(flat_cube, axis=(1, 2)) - read_noise2

    flat_var_arr_sort_args = np.argsort(flat_var_arr)
    flat_var_arr = flat_var_arr[flat_var_arr_sort_args]
    flat_mean_arr = flat_mean_arr[flat_var_arr_sort_args]

    gain_cal = np.mean(flat_mean_arr) / np.var(flat_mean_arr)
    # gain_cal = np.mean(gain_cal_arr)
    # gain_std = np.std(gain_cal_arr)

    # # fit the gain
    # def func(x, a):
    #     return a*x

    # def residual(a, x_data, y_data):
    #     return func(x_data, a) - y_data

    # gain_fit_arr, gain_cov = leastsq(residual, x0=1, args=(flat_var_arr, flat_mean_arr))
    # gain_fit = gain_fit_arr[0]

    # flat_mean_fit = func(flat_var_arr, gain_fit)

    # chi2 = np.sum(np.power(flat_mean_arr - flat_mean_fit, 2))
    # dof = flat_mean_fit.shape[0]
    # chi2dof = chi2 / dof

    # if save_fig:
    #     plt.figure()
    #     plt.title('Gain ' + r'$\sigma_{\rm{Gain}}=$' + f'{gain_std:.3g}')
    #     plt.plot(flat_var_arr, flat_mean_arr, 'k.', label='data')
    #     plt.plot(flat_var_arr, flat_var_arr * gain_cal, 'r--', label=f'gain={gain_cal:.3g}')
    #     plt.ylabel(r'MEAN $N_{e^-} / \rm{Gain}$')
    #     plt.xlabel(r'VAR $N_{e^-} / \rm{Gain}^2$')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.savefig(save_fig, dpi=300)
    #     plt.close()
    
    return gain_cal, np.sqrt(read_noise2 * gain_cal**2)


def cal_gain_using_flat(flat_cube: np.ndarray, bias_var: np.ndarray, size_subimg: int, save_perfix: str=''):
    assert flat_cube.shape[1:] == bias_var.shape, 'flat_cube shape [1,2] shoud be same as bias_var shape [0,1]'
    N_flat, w_flat, h_flat = flat_cube.shape
    size_subimg = 10
    flat_stat_dict = {
        'w_st': [], 
        'w_ed': [], 
        'h_st': [], 
        'h_ed': [], 
        'gain': [], 
        'read_noise': [], 
    }
    img_save_dir = BASE / 'figure' / 'gain'
    img_save_dir.mkdir(exist_ok=True, parents=True)
    data_save_dir = BASE / 'my_data' / 'gain'
    data_save_dir.mkdir(exist_ok=True, parents=True)
    N_cal_w = w_flat // size_subimg
    N_cal_h = h_flat // size_subimg
    N_cal_total = N_cal_w * N_cal_h
    for idx in tqdm(
        range(N_cal_total), 
        desc='cal gain', 
        total=N_cal_total,
    ):
        i = idx // N_cal_h
        j = idx % N_cal_h
        i_st, i_ed = i*size_subimg, (i+1)*size_subimg
        j_st, j_ed = j*size_subimg, (j+1)*size_subimg
        # save_path = img_save_dir / f'{save_perfix}{i_st:04d}_{i_ed:04d}_{j_st:04d}_{j_ed:04d}.png'
        gain_cal, read_noise = cal_gain(flat_cube[:, i_st:i_ed, j_st:j_ed], bias_var[i_st:i_ed, j_st:j_ed])
        # gain_fit, sig_gain, read_noise2, flat_var_arr, flat_mean_arr = cal_gain(flat_cube[:, i_st:i_ed, j_st:j_ed], bias_var[i_st:i_ed, j_st:j_ed], save_fig=save_path)
        flat_stat_dict['w_st'].append(i_st)
        flat_stat_dict['w_ed'].append(i_ed)
        flat_stat_dict['h_st'].append(j_st)
        flat_stat_dict['h_ed'].append(j_ed)
        flat_stat_dict['gain'].append(gain_cal)
        flat_stat_dict['read_noise'].append(read_noise)
    gain_arr = np.array(gain_lst)
    read_noise_arr = np.array(read_noise_lst)
    flat_var_arr = np.array(flat_var_lst)
    flat_mean_arr = np.array(flat_mean_lst)
    gain_mean = np.mean(gain_arr)
    gain_std = np.std(gain_arr)
    read_noise_mean = np.mean(read_noise_arr)
    read_noise_std = np.std(read_noise_arr)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(flat_var_arr, flat_mean_arr, 'k.', label='data')
    # ax.plot(flat_var_arr, flat_var_arr * gain_mean, 'r-', alpha=0.8, label=f'gain={gain_mean:.3g}')
    ax.set_xlabel(r'VAR $N_{e^-} / \rm{Gain}^2$')
    ax.set_ylabel(r'MEAN $N_{e^-} / \rm{Gain}$')
    ax.set_title('Gain ' + r'$\sigma_{\rm{Gain}}=$' + f'{gain_std:.3g}')
    ax.grid(True)
    ax.legend(loc='upper left')
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    df_gain = pd.DataFrame(flat_stat_dict)
    df_gain.to_csv(data_save_dir / f'{save_perfix}{i_st:04d}_{i_ed:04d}_{j_st:04d}_{j_ed:04d}.csv', index=False)

    gain_cal = np.mean(df_gain.gain)
    gain_std = np.std(df_gain.gain)
    read_noise_mean = np.mean(df_gain.read_noise)
    read_noise_std = np.std(df_gain.read_noise)

    return gain_cal, gain_std, read_noise_mean, read_noise_std


def check_gain(bias_fpath_lst: list, flat_fpath_lst: list, perfix: str='', kernel_size:int=10, interest_area: tuple=(4950, 5000, 4950, 5000)):
    print('----------------------------')
    print('job prefix: ', perfix)
    print('----------------------------')
    # bias_mask = gen_shape_mask(bias_fpath_lst, target_shape=(6388, 9576), tag='bias')
    # bias_fpath_lst = [bias_fpath_lst[i] for i in range(len(bias_fpath_lst)) if bias_mask[i]]
    # print(f'\t avaliable bias: {len(bias_fpath_lst):10d}')
    assert len(bias_fpath_lst) > 0

    # flat_mask = gen_shape_mask(flat_fpath_lst, target_shape=(6388, 9576), tag='flat')
    # flat_fpath_lst = [flat_fpath_lst[i] for i in range(len(flat_fpath_lst)) if flat_mask[i]]
    # print(f'\t avaliable flat: {len(flat_fpath_lst):10d}')
    assert len(flat_fpath_lst) > 0
    
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
    bias_var = np.nanvar(bias_cube, axis=0) # read noise
    del bias_cube
    print(f'stacked bias shape: {bias.shape}')

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

    h_st, h_ed, w_st, w_ed = interest_area
    res = cal_gain_using_flat(
        flat_cube[:, h_st:h_ed, w_st:w_ed], 
        bias_var[h_st:h_ed, w_st:w_ed], 
        kernel_size, 
        save_perfix=perfix + f'{h_st}_{h_ed}_{w_st}_{w_ed}'
    )
    print('job done!')
    print('----------------------------')
    print('')
    return res


def bias_flat_finder(data_dir: Path, target_shape=(6388, 9576)):
    assert data_dir.exists()
    data_subdir_lst = list(filter(lambda d: d.is_dir(), data_dir.glob('*')))
    fits_map = {}
    for idx, this_dir in enumerate(data_subdir_lst):
        print(f'now finding dir {idx+1}/{len(data_subdir_lst)}')
        try:
            bias_fpath_lst = list(this_dir.glob('*-bias-*.fit'))
            bias_mask = gen_shape_mask(bias_fpath_lst, target_shape=target_shape, tag='bias')
            bias_fpath_lst = [bias_fpath_lst[i] for i in range(len(bias_fpath_lst)) if bias_mask[i]]
            assert len(bias_fpath_lst) > 1, 'at least 2 bias required: ' + this_dir.as_posix()

            flat_fpath_lst = list(this_dir.glob('*-flat-*.fit'))
            flat_mask = gen_shape_mask(flat_fpath_lst, target_shape=target_shape, tag='flat')
            flat_fpath_lst = [flat_fpath_lst[i] for i in range(len(flat_fpath_lst)) if flat_mask[i]]
            assert len(flat_fpath_lst) > 0, 'at least 1 flat required: ' + this_dir.as_posix()

            fits_map[this_dir] = {'bias': bias_fpath_lst, 'flat': flat_fpath_lst}
        except:
            continue
    return fits_map


def main():
    file_finder_dict_cache_path = CACHEDIR / 'bias_flat_map.cache'
    if file_finder_dict_cache_path.exists():
        print('file_find_map found')
        with open(file_finder_dict_cache_path, 'rb') as fid:
            file_find_map = pickle.load(fid)
    else:
        file_find_map = bias_flat_finder(DATADIR, target_shape=(6388, 9576))
        with open(file_finder_dict_cache_path, 'wb') as fid:
            pickle.dump(file_find_map, fid)
    
    res_map = {
        'perfix': [], 
        'gain': [], 
        'gain_std': [], 
        'read_noise': [], 
        'read_noise_std': [], 
    }
    
    for key, val in file_find_map.items():
        bias_fpath_lst = val['bias']
        flat_fpath_lst = val['flat']
        if key.stem.startswith('mst1'):
            interest_area = (3000, 3050, 5000, 5050)
            kernel_size = 10
        elif key.stem.startswith('st001'):
            interest_area = (4950, 5000, 4950, 5000)
            kernel_size = 10
        else:
            raise RuntimeError('un-supported input dir perfix ' + key.stem)
        gain, gain_std, read_noise, read_noise_std = check_gain(
            bias_fpath_lst, flat_fpath_lst, 
            perfix=key.stem + '_', 
            kernel_size=kernel_size, 
            interest_area=interest_area, 
        )
        print(key.stem, gain, gain_std)
        res_map['perfix'].append(key.stem)
        res_map['gain'].append(gain)
        res_map['gain_std'].append(gain_std)
        res_map['read_noise'].append(read_noise)
        res_map['read_noise_std'].append(read_noise_std)
    
    df = pd.DataFrame(res_map)
    print(df)
    df.to_csv(BASE / 'my_data' / 'gain' / 'cal0.csv', index=False)

    df_mst = df[df.perfix.agg(lambda x: x.startswith('mst'))]
    df_st = df[df.perfix.agg(lambda x: x.startswith('st'))]

    def prefix_2_mjd(pre):
        if pre.startswith('mst'):
            date_str = pre.split('_')[1]
        elif pre.startswith('st'):
            date_str = pre.split('-')[1]
        else:
            raise RuntimeError()
        date_iso = '-'.join([date_str[:4], date_str[4:6], date_str[6:8]])
        date_time = Time(date_iso)
        return date_time.to_value('mjd', subfmt='float')

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.errorbar(
        x=[prefix_2_mjd(v) for v in df_mst.perfix.to_list()],
        y=df_mst.gain, 
        yerr=df_mst.gain_std, 
        fmt='ko', 
        linewidth=0, 
        elinewidth=1, 
        linestyle='none', 
        label='mst', 
    )
    ax.errorbar(
        x=[prefix_2_mjd(v) for v in df_st.perfix.to_list()],
        y=df_st.gain, 
        yerr=df_st.gain_std, 
        fmt='ks', 
        linewidth=0, 
        elinewidth=1, 
        linestyle='none', 
        label='st', 
    )
    ax2 = ax.twinx()
    ax2.errorbar(
        x=[prefix_2_mjd(v) for v in df_mst.perfix.to_list()],
        y=df_mst.read_noise, 
        yerr=df_mst.read_noise_std, 
        fmt='bo', 
        linewidth=0, 
        elinewidth=1, 
        linestyle='none', 
        label='mst', 
    )
    ax2.errorbar(
        x=[prefix_2_mjd(v) for v in df_st.perfix.to_list()],
        y=df_st.read_noise, 
        yerr=df_st.read_noise_std, 
        fmt='bs', 
        linewidth=0, 
        elinewidth=1, 
        linestyle='none', 
        label='st', 
    )
    # ax2.plot(df_mst.perfix.agg(lambda x: prefix_2_mjd(x)), df_mst.read_noise, 'bo')
    # ax2.plot(df_st.perfix.agg(lambda x: prefix_2_mjd(x)), df_st.read_noise, 'bs')
    ax2.set_ylabel(r'$N_{\rm RD} (e^-/\rm{pixel})$')
    ax.set_xlabel('JD')
    ax.set_ylabel(r'$\rm{Gain} (e^-/\rm{ADU})$')
    ax.legend(loc='best')
    fig.savefig(BASE / 'figure' / 'my_gain.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()