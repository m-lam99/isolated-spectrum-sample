#Function definitions to calculate FstarN
import numpy as np
from astropy import constants as C
from scipy.interpolate import interp1d


def transit_depth(r, Rp_Rs, u1, u2):
    """Calculate the transit depth according to the quadratic
    limb darkening law (Csizmadia et al. 2013)
    """
    mu = np.sqrt(1 - r**2)
    D = Rp_Rs**2 * (
        (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2) / (1 - u1 / 3 - u2 / 6)
    )
    return D

def limb_darkening(r, Rp_Rs, u1, u2):
    """Calculate the transit light curve
    """
    mask = -1 * (np.floor(r) - 1)
    D = transit_depth(r,Rp_Rs,u1,u2)* mask
    return 1 - D

def compute_spectra_LD(F_norm, r, Rp_Rs, u1, u2):
    """Incorporate limb darkening into continuum normalised spectra,
    following Cegla et al. 2016
    """
    return F_norm * limb_darkening(r, Rp_Rs, u1, u2)[:, np.newaxis]

def compute_residuals(F,F_oot):
    """Compute residuals from normalised spectra
    
    Parameters:
    ----------
    Continuum normalised flux with limb darkening
    """
    res = F/F_oot
    return res

def f_obsc(res, D, F_oot):
    """Compute obscured flux for every exposure
    """
    return (1 - res) / D[:, np.newaxis] * F_oot

def inverse_shift_SR(F_obsc, in_transit_idx, wl_base, SR):
    F_obsc_interp = np.empty_like(F_obsc[in_transit_idx.flatten(),:])
    for i, idx in enumerate(in_transit_idx):

        mask = SR.masks[idx[0]].copy()
        mask[SR.masks[idx[0]] == 1] = 0
        mask[np.isnan(SR.masks[idx[0]])] = 1
        mask[mask == 0] = np.nan
        masked_vel_grid = mask * SR.vel_grid

        v = np.nanmean(masked_vel_grid)

        beta = v / C.c.value
        shift = np.sqrt((1 + beta) / (1 - beta))

        wl = SR.wl / shift

        fr_interp_temp = interp1d(wl, F_obsc[idx[0], :], bounds_error=False)(wl_base)
        F_obsc_interp[i] = fr_interp_temp
        
    return F_obsc_interp

def inverse_shift(F_obsc, in_transit_idx, wl_base, xp, vsini):    
    F_obsc_interp = np.empty_like(F_obsc[in_transit_idx.flatten(),:])
    for i, idx in enumerate(in_transit_idx):

        v = xp[idx] * vsini
        beta = v / C.c.value
        doppler_shift = np.sqrt((1 + beta) / (1 - beta))

        wl = wl_base / doppler_shift

        interp_temp = interp1d(wl, F_obsc[idx[0], :], bounds_error=False)(wl_base)
        F_obsc_interp[i] = interp_temp
        
    return F_obsc_interp

def flux_intensity(r, u1, u2):
    """Compute limb-darkened intensity across the stellar disc
    """
    mu = np.sqrt(1 - r**2)
    I = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
    return I

def weighted_average(F_obsc, r, Rp_Rs, u1, u2, in_transit_idx):
    """Compute the weighted average of all F_obsc in transit
    """
    r_outside = r + Rp_Rs
    r_inside = r - Rp_Rs

    A_center = np.pi * np.nanmin(r_inside) ** 2
    I_0 = flux_intensity(np.nanmin(r), u1, u2)
    Fn_centre = F_obsc[np.nanargmin(r_inside)] * A_center * I_0
    weight_centre = A_center * I_0

    areas = np.pi * (r_outside ** 2 - r_inside ** 2)
    I = flux_intensity(r,u1,u2)
    Fn = F_obsc * areas[:,np.newaxis] * I[:,np.newaxis]
    weight = areas * I

    # Check which rows are entirely nans and get rid of them!
    Fn = Fn[np.argwhere(np.nansum(Fn,axis=1)!=0).flatten(),:]
    weight = weight[np.argwhere(np.nansum(Fn,axis=1)!=0).flatten()]
    
    FstarN = np.nansum(np.vstack((Fn, Fn_centre)), axis=0) / np.nansum(
        np.hstack((weight.flatten(), weight_centre))
    )
    
    return FstarN

def compute_FstarN_SR(F,xp,yp,Rp_Rs,u1,u2,SR):
    r = np.sqrt(xp**2+yp**2)
    out_of_transit_idx = np.argwhere(r > 1)
    in_transit_idx = np.argwhere(r < 1 - Rp_Rs)
    
    F_oot = np.nanmean(F[out_of_transit_idx],axis=0).flatten()
    # F_LD = compute_spectra_LD(F_norm, r, Rp_Rs,u1,u2)
    
    res = compute_residuals(F,F_oot)
    D = transit_depth(r,Rp_Rs,u1,u2)
    F_obsc = f_obsc(res,D, F_oot)
    F_obsc /= np.nanmedian(F_obsc,axis=0) # normalise to same level before averaging
    
    F_obsc_shifted = inverse_shift_SR(F_obsc,in_transit_idx, SR.wl, SR)
    FstarN = weighted_average(F_obsc_shifted,r[in_transit_idx.flatten()],Rp_Rs,u1,u2,in_transit_idx)
    
    return FstarN

def compute_FstarN(wl,F,xp,yp,Rp_Rs,u1,u2,vsini, in_transit_idx, out_of_transit_idx):
    r = np.sqrt(xp**2+yp**2)
    
    F_oot = np.nanmean(F[out_of_transit_idx],axis=0).flatten()
    # F_LD = compute_spectra_LD(F, r, Rp_Rs,u1,u2)
    
    res = compute_residuals(F,F_oot)
            
    D = transit_depth(r,Rp_Rs,u1,u2)
    F_obsc = f_obsc(res,D, F_oot)
    
    F_obsc_shifted = inverse_shift(F_obsc, in_transit_idx, wl,xp,vsini)
    FstarN = weighted_average(F_obsc_shifted,r[in_transit_idx.flatten()],Rp_Rs,u1,u2,in_transit_idx)
    
    return FstarN

def compute_FstarN_MX(wl,F,F_oot_HARPS,xp,yp,Rp_Rs,u1,u2,vsini, in_transit_idx, out_of_transit_idx):
    r = np.sqrt(xp**2+yp**2)
    
    # F_LD = compute_spectra_LD(flux, r, Rp_Rs,u1,u2)
    F_oot = np.nanmean(F[out_of_transit_idx],axis=0).flatten()

    res = compute_residuals(F,F_oot)
            
    D = transit_depth(r,Rp_Rs,u1,u2)
    F_obsc = f_obsc(res,D, F_oot_HARPS)
    
    F_obsc_shifted = inverse_shift(F_obsc, in_transit_idx, wl,xp,vsini)
    FstarN = weighted_average(F_obsc_shifted,r[in_transit_idx.flatten()],Rp_Rs,u1,u2,in_transit_idx)
    
    return FstarN