import numpy as np
from scipy.interpolate import interp1d


class Obs():
    def __init__(self, 
                 pixel_size=10e-6, # m/pixel
                 pix_scale=0.6, # arcsec/pixel
                 noise_read=4, # e-/pixel
                 dark_signal=4/3600.0, # e-/pixel/s
                 mag_bk=21, # mag/arcsec^2
                 k=0.35, # mag/airmass
                 airmass=1.2,
                 tel_eff=0.4, 
                 seeing=1.5, # arcsec
                ):
        self.pixel_size = pixel_size
        self.pix_scale = pix_scale
        self.noise_read = noise_read
        self.dark_signal = dark_signal
        self.mag_bk = mag_bk
        self.k = k
        self.airmass = airmass
        self.tel_eff = tel_eff
        self.seeing = seeing

    def set_qe(self, wave_arr, qe_arr):
        func_qe = interp1d(wave_arr, qe_arr, kind='linear')
        self.func_qe = func_qe

    def cal_mag_limit(self, exp_time, wavelength, bandwidth, snr):
        pass