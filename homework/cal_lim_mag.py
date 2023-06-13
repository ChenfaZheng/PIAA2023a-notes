import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm
from astropy import units as u
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pathlib import Path
BASE = Path(__file__).parent.parent


class CCD():
    def __init__(
            self, 
            pix_size=10e-6 * u.meter, # m/pixel
            pix_scale=0.6 * u.arcsec, # arcsec/pixel
            noise_read=4 * u.electron, # e-/pixel
            dark_signal=4/3600.0 * u.second**-1 * u.electron, # e-/pixel/s
    ):
        self.pix_size = pix_size
        self.pix_scale = pix_scale
        self.noise_read = noise_read
        self.dark_signal = dark_signal
    
    def set_response(self, wave_arr, qe_arr):
        self.func_qe = lambda l: np.interp(l, wave_arr, qe_arr)
    
    def get_qe(self, wavelength):
        assert hasattr(self, 'func_qe'), "QE function not set"
        return self.func_qe(wavelength)
    
    def go_through(self, energy, wavelength):
        # flux: J / s / m^2, not per frequency
        E_photon = (6.626e-34 * u.joule * u.second) * (3e8 * u.meter / u.second) / wavelength / u.photon
        n_photon = energy / E_photon
        this_qe = self.get_qe(wavelength)
        n_electron = n_photon * this_qe
        return n_electron


class Telescope():
    def __init__(
            self, 
            tel_eff=0.4,    # telescope efficiency
            D=1.0 * u.meter, # telescope diameter
    ):
        self.tel_eff = tel_eff
        self.D = D

    def get_sep_lim(self, wavelength):
        return 1.22 * wavelength / self.D * u.radian
    
    def go_through(self, flux, aperture, fwhm):
        area_in = np.pi * (self.D / 2) ** 2
        energy_in = flux * area_in
        psf_ratio = 1 - np.exp(-aperture**2 / fwhm**2)
        energy_out = energy_in * psf_ratio
        return energy_out


class Observe():
    def __init__(
            self, 
            exp_time=30 * u.second, # exposure time
            seeing=1.5 * u.arcsec, # seeing
            k=0.35 * u.mag, # mag/airmass
            airmass=1.2,    # airmass
    ):
        self.exp_time = exp_time
        self.seeing = seeing
        self.k = k
        self.airmass = airmass


class Event():
    def __init__(
        self, 
        mag_obj=20 * u.mag, # object magnitude
        mag_sky=21 * u.mag, # sky magnitude
        wavelength=500 * u.nm, # wavelength
        bandwidth=100 * u.nm, # bandwidth
    ):
        self.mag_obj = mag_obj
        self.mag_sky = mag_sky
        self.wavelength = wavelength
        self.bandwidth = bandwidth
        self.flux_obj = None
        self.flux_sky = None
    
    def set_mag_obj(self, mag_obj):
        self.mag_obj = mag_obj

    
    def _cal_flux(self, mag, k, airmass, eff):
        flux_nu = 10**(-0.4 * (mag + k * airmass - 8.9 * u.mag) / u.mag) * eff * u.Jy
        flux = flux_nu * (3e8 * u.meter / u.second) / self.wavelength**2 * self.bandwidth
        return flux


    def cal_flux(
            self, 
            tele: Telescope,
            obs: Observe,
    ):
        self.flux_obj = self._cal_flux(
            self.mag_obj, 
            obs.k, 
            obs.airmass, 
            tele.tel_eff,
        )
        self.flux_sky = self._cal_flux(
            self.mag_sky,
            obs.k,
            obs.airmass,
            tele.tel_eff,
        )
        return self.flux_obj, self.flux_sky


class Observertory():
    def __init__(self):
        pass

    def set_ccd(self, ccd: CCD):
        self.ccd = ccd

    def set_telescope(self, telescope: Telescope):
        self.telescope = telescope

    def set_obs(self, obs: Observe):
        self.obs = obs

    def cal_mag_limit(self, event: Event, snr=None, r_aperture=None, plot=False, save=None):
        if r_aperture is None:
            r_aperture = 2 * u.arcsec
        else:
            try:
                r_aperture = r_aperture.to(u.arcsec)
            except:
                raise ValueError("r_aperture should be astropy quantity with unit of angle")
        if snr is None:
            snr = 5
        
        mag_min = 0 * u.mag
        mag_max = 30 * u.mag
        mag_arr = np.linspace(mag_min, mag_max, 100)
        snr_arr = np.empty(mag_arr.shape)
        for i, mag in enumerate(mag_arr):
            event.set_mag_obj(mag)
            snr_arr[i] = self.cal_snr(event, r_aperture=r_aperture)
        func_snr = interp1d(mag_arr, snr_arr)
        func_snr_scaler = lambda x: func_snr(x * u.mag) - snr
        mag_limit = brentq(func_snr_scaler, mag_min.value, mag_max.value)
        if plot:
            fig, ax = plt.subplots()
            ax.semilogy(mag_arr, snr_arr)
            ax.axhline(snr, color='k', ls='--')
            ax.axvline(mag_limit, color='k', ls='--')
            ax.set_xlabel('mag')
            ax.set_ylabel('SNR')
            ax.set_title(f'SNR = {snr:.2f}')
            if save is not None:
                fig.savefig(save)
            plt.close(fig)
        return mag_limit


    def cal_snr(self, event: Event, r_aperture=None):
        # compare seeing and telescope resolution
        sep_lim_tele = self.telescope.get_sep_lim(event.wavelength)
        sep_lim = max(sep_lim_tele, self.obs.seeing)
        # define r_aperture
        if r_aperture is None:
            r_aperture = 1.5 * sep_lim / 2.0
        else:
            try:
                r_aperture = r_aperture.to(u.arcsec)
            except:
                raise ValueError("r_aperture should be astropy quantity with unit of angle")
        size_aperture = r_aperture / self.ccd.pix_scale * self.ccd.pix_size
        # count pixel number in aperture
        n_pix = np.pi * r_aperture**2 / self.ccd.pix_scale**2
        # area of aperture in m^2
        area_aperture = n_pix * self.ccd.pix_size**2

        # calculate flux from sky and star
        flux_obj, flux_sky = event.cal_flux(
            tele=self.telescope,
            obs=self.obs,
        )
        # calculate flux from sky and star in aperture
        energy_focal_obj = self.telescope.go_through(flux_obj, r_aperture, sep_lim)
        energy_focal_sky = self.telescope.go_through(flux_sky, r_aperture, sep_lim)
        # calculate flux from sky and star in aperture in e-/s
        ne_obj = self.ccd.go_through(energy_focal_obj, event.wavelength)
        ne_sky = self.ccd.go_through(energy_focal_sky, event.wavelength)
        # calculate noise
        ne_read = self.ccd.noise_read
        ne_dark = self.ccd.dark_signal
        # consider exposure time
        N_obj = ne_obj * self.obs.exp_time
        N_sky = ne_sky * self.obs.exp_time
        N_read = ne_read
        N_dark = ne_dark * self.obs.exp_time
        # dimensionless
        N_obj = N_obj.to(u.electron).value
        N_sky = N_sky.to(u.electron).value
        N_read = N_read.to(u.electron).value
        N_dark = N_dark.to(u.electron).value
        # calculate snr
        snr = N_obj / np.sqrt(N_obj + N_sky + n_pix *( N_read**2 + N_dark ))
        return snr


def main():
    ccd = CCD(
        pix_size=10e-6 * u.meter, # m/pixel
        pix_scale=0.6 * u.arcsec, # arcsec/pixel
        noise_read=4 * u.electron, # e-/pixel
        dark_signal=4/3600.0 * u.second**-1 * u.electron, # e-/pixel/s
    )
    ccd.set_response(
        wave_arr=np.array([250, 400, 500, 650, 900]) * u.nm,
        qe_arr=np.array([0.3, 0.75, 0.75, 0.8, 0.5]) * u.electron / u.photon, 
    )
    tele = Telescope(
        tel_eff=0.4,    # telescope efficiency
        D=1.0 * u.meter, # telescope diameter
    )
    obs = Observe(
        exp_time=20 * u.second, # exposure time
        seeing=1.5 * u.arcsec, # seeing
        k=0.35 * u.mag, # mag/airmass
        airmass=1.2,    # airmass
    )
    obsertory = Observertory()
    obsertory.set_ccd(ccd)
    obsertory.set_telescope(tele)
    obsertory.set_obs(obs)

    event = Event(
        mag_obj=20 * u.mag, # object magnitude
        mag_sky=21 * u.mag, # sky magnitude
        wavelength=500 * u.nm, # wavelength
        bandwidth=100 * u.nm, # bandwidth
    )
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel("aperture radius (arcsec)")
    ax.set_ylabel("SNR")
    ax.set_title("SNR vs aperture radius")
    for r_aperture in np.arange(0.1, 5, 0.1) * u.arcsec:
        snr = obsertory.cal_snr(event, r_aperture)
        ax.plot(r_aperture, snr, "k.")
    # plt.show()
    plt.savefig(BASE / 'homework'/ "snr_vs_aperture_radius.png", dpi=300)
    plt.close()

    obsertory.cal_mag_limit(
        event=event,
        snr=5,
        r_aperture=1.5 * u.arcsec,
        plot=True,
        save=BASE / 'homework'/ "mag_limit.png",
    )


if __name__ == '__main__':
    main()