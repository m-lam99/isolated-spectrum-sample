# Function definitions to calculate FstarN
import numpy as np
from astropy import constants as C
from scipy.interpolate import interp1d


def flux_intensity(r, u1, u2):
    """Compute limb-darkened intensity across the stellar disc
    according to the quadratic limb darkening law (Kipping 2013)
    which describes the intensity as:
    I(mu)/I(1) = 1 - u1(1-mu) - u2(1-mu)^2

    Parameters:
    -----------
    r : 1d np.array(float)
        Radial distance from centre of star, normalised to radius
        of star.
    u1, u2 : float
        Limb darkening coefficients according to quadratic limb
        darkening law.

    Returns:
    I : 1d np.array(float)
        Relative flux intensity at position defined by r.
    """
    mu = np.sqrt(1 - r**2)
    I = 1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2
    return I


def transit_depth(r, Rp_Rs, u1, u2):
    """Calculate the transit depth according to the quadratic
    limb darkening law (Csizmadia et al. 2013).
    The quadratic limb darkening law is based on (Kipping 2013)

    Parameters:
    -----------
    r : 1d np.array(float)
        Radial distance from centre of star, normalised to radius
        of star.
    Rp_Rs : float
        Ratio of planetary radius to stellar radius.
    u1, u2 : float
        Limb darkening coefficients according to quadratic limb
        darkening law.

    Returns:
    --------
    D : 1d np.array(float)
        Depth of transit light curve at position defined by r
    """
    mu = np.sqrt(1 - r**2)
    D = Rp_Rs**2 * ((1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2) / (1 - u1 / 3 - u2 / 6))
    return D


def light_curve(r, Rp_Rs, u1, u2):
    """Calculate the transit light curve from the transit depth

    Parameters:
    r : 1d np.array(float)
        Radial distance from centre of star, normalised to radius
        of star.
    Rp_Rs : float
        Ratio of planetary radius to stellar radius.
    u1, u2 : float
        Limb darkening coefficients according to quadratic limb
        darkening law.

    Returns:
    --------
    1 - D : 1d np.array(float)
        Transit light curve at position defined by r.
    """
    mask = -1 * (np.floor(r) - 1)
    D = transit_depth(r, Rp_Rs, u1, u2) * mask
    return 1 - D


def compute_residuals(F, F_oot):
    """Compute residuals from normalised time-series spectra.

    Parameters:
    ----------
    F : 2d np.array(float)
        Continuum normalised flux with limb darkening.
    F_oot : 1d np.array(float)
        Mean out-of-transit stellar spectrum.

    Returns:
    --------
    res : 2d np.array(float)
        Residual spectrum
    """
    res = F / F_oot
    return res


def compute_obscured_flux(res, D, F_oot):
    """Compute obscured flux for every exposure according to Eqn. 7 in
    Lam et al. (in prep)

    Parameters:
    -----------
    res : 2d np.array(float)
        Residual spectrum
    D : 1d np.array(float)
        Transit depth
    F_oot : 1d np.array(float)
        Out-of-transit stellar spectrum

    Returns:
    --------
    F_obsc : 2d np.array(float)
        Obscured flux at every exposure
    """
    F_obsc = (1 - res) / D[:, np.newaxis] * F_oot
    return F_obsc


def inverse_shift_SR(F_obsc, in_transit_idx, wl_base, SR):
    """Shift the obscured spectrum back to the rest frame.

    Calculates the inverse Doppler shift and applies it to the obscured
    spectrum. This function is only applicable to the StarRotator-generated
    spectrum, where the intial Doppler shift is calculated by the mask
    over the grid cells.

    Parameters:
    -----------
    F_obsc : 2d np.array(float)
        Obscured flux at every exposure.
    in_transit_idx:  1d np.array(int)
        Indices determining which exposure is taken during an in-transit
        event.
    wl_base : 1d np.array(float)
        Wavelength array on which to interpolate the final spectrum.
    SR : StarRotator object
        StarRotator object used to model synthetic spectra.

    Returns:
    --------
    F_obsc_interp : 2d np.array(float)
        Obscured spectra corrected for the Doppler shift.
    """
    F_obsc_interp = np.empty_like(F_obsc[in_transit_idx.flatten(), :])
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
    """Shift the obscured spectrum back to the rest frame.

    Calculates the inverse Doppler shift and applies it to the obscured
    spectrum.

    Parameters:
    -----------
    F_obsc : 2d np.array(float)
        Obscured flux at every exposure.
    in_transit_idx : 1d np.array(int)
        Indices determining which exposure is taken during an in-transit
        event.
    wl_base : 1d np.array(float)
        Wavelength array on which to interpolate the final spectrum.
    xp : 1d np.array(float)
        The x-coordinate of the orthogonal distance from the spin-axis of
        the planet for each exposure. This is calculated using (Cegla et al. 2016)
    vsini : float
        Projected rotational velocity of the host star.

    Returns:
    --------
    F_obsc_interp : 2d np.array(float)
        Obscured spectra corrected for the Doppler shift.
    """
    F_obsc_interp = np.empty_like(F_obsc[in_transit_idx.flatten(), :])
    for i, idx in enumerate(in_transit_idx):

        v = xp[idx] * vsini
        beta = v / C.c.value
        doppler_shift = np.sqrt((1 + beta) / (1 - beta))

        wl = wl_base / doppler_shift

        interp_temp = interp1d(wl, F_obsc[idx[0], :], bounds_error=False)(wl_base)
        F_obsc_interp[i] = interp_temp

    return F_obsc_interp


def weighted_average(F_obsc, r, Rp_Rs, u1, u2):
    """Compute the weighted average of all F_obsc in transit to determine
    the final isolated spectrum, F_{\star,N}. Each spectrum is weighted
    according to the annuluar area defined by the position and radius
    of the planet, and also the stellar flux intensity corresponding to
    the position of the planet.

    Parameters:
    -----------
    F_obsc: 2d np.array(float)
        Obscured spectra corrected for the Doppler shift.
    r : 1d np.array(float)
        Radial distance from centre of star, normalised to radius
        of star.
    Rp_Rs : float
        Ratio of planetary radius to stellar radius.
    u1, u2 : float
        Limb darkening coefficients according to quadratic limb
        darkening law.

    Returns:
    --------
    FstarN : 1d np.array(float)
        Weighted average of all obscured spectra.
    """
    r_outside = r + Rp_Rs  # Outer radius of annulus
    r_inside = r - Rp_Rs  # Inner radius of annulus

    # Assume central disc has the same spectrum as the inner-most annulus
    A_center = np.pi * np.nanmin(r_inside) ** 2
    I_0 = flux_intensity(np.nanmin(r), u1, u2)
    Fn_centre = F_obsc[np.nanargmin(r_inside)] * A_center * I_0
    weight_centre = A_center * I_0

    # Compute area and intensity weights for all spectra
    areas = np.pi * (r_outside**2 - r_inside**2)
    I = flux_intensity(r, u1, u2)
    Fn = F_obsc * areas[:, np.newaxis] * I[:, np.newaxis]
    weight = areas * I

    # Check which rows are entirely nans and get rid of them!
    Fn = Fn[np.argwhere(np.nansum(Fn, axis=1) != 0).flatten(), :]
    weight = weight[np.argwhere(np.nansum(Fn, axis=1) != 0).flatten()]

    # Weighted average equation
    FstarN = np.nansum(np.vstack((Fn, Fn_centre)), axis=0) / np.nansum(
        np.hstack((weight.flatten(), weight_centre))
    )

    return FstarN


def compute_FstarN(
    wl,
    F,
    xp,
    yp,
    Rp_Rs,
    u1,
    u2,
    vsini,
    in_transit_idx,
    out_of_transit_idx,
    instrument="",
    SR=None,
    F_oot_HARPS=None,
):
    """Compute isolated narrow spectrum from time-series spectral observations.
    This is intended for fast-rotating A-type stars with a known exoplanet with
    rotation-broadened spectra. This follows the methodology presented in Lam
    et al. (in prep)

    Parameters:
    -----------
    wl : 1d np.array(float)
        Wavelength of each observation, shifted to the correct rest frame.
    F : 2d np.array(float)
        Time-series spectral observations of the target star.
    xp, yp : 1d np.array(float)
        The x- and y-coordinate of the orthogonal distance from the spin-axis of
        the planet for each exposure. This is calculated using (Cegla et al. 2016)
    Rp_Rs : float
        Ratio of planetary radius to stellar radius.
    u1, u2 : float
        Limb darkening coefficients according to quadratic limb
        darkening law.
    vsini : float
        Projected rotational velocity of the host star.
    in_transit_idx : 1d np.array(int)
        Indices determining which exposure is taken during an in-transit event.
    out_of_transit_idx : 1d np.array(int)
        Indices determining which exposure is taken during an out-of-transit
        event.
    instrument : str
        Which mode, or instrument the spectra is from. Currently, options are
        HARPS, StarRotator or MAROON-X.
    SR : StarRotator object
        StarRotator object used to model synthetic spectra. Only used if
        instrument = "StarRotator".
    F_oot_HARPS : 1d np.array(float)
        Out-of-transit broadened spectrum measured from HARPS. Only used if
        instrument = "MAROON-X" since we don't have 1d spectra from MAROON-X.

    Returns:
    --------
    FstarN : 1d np.array(float)
        Isolated narrow spectrum
    """
    r = np.sqrt(xp**2 + yp**2)

    F_oot = np.nanmean(F[out_of_transit_idx], axis=0).flatten()
    F_LD = F * light_curve(r, Rp_Rs, u1, u2)[:, np.newaxis]

    res = compute_residuals(F_LD, F_oot)
    D = transit_depth(r, Rp_Rs, u1, u2)

    # Depending on which "mode" or instrument the spectra is from, slightly
    # different calculations are used.
    if instrument == "HARPS":
        F_obsc = compute_obscured_flux(res, D, F_oot)
        F_obsc_shifted = inverse_shift(F_obsc, in_transit_idx, wl, xp, vsini)

    elif instrument == "StarRotator":
        if SR == None:
            raise Exception(
                "StarRotator mode has been selected but no StarRotator "
                "object has been input. Set the StarRotator object."
            )
        F_obsc = compute_obscured_flux(res, D, F_oot)
        F_obsc_shifted = inverse_shift_SR(F_obsc, in_transit_idx, SR.wl, SR)

    elif instrument == "MAROON-X":
        if F_oot_HARPS == None:
            raise Exception(
                "MAROON-X instrument has been selected but no reference "
                "out-of-transit stellar spectrum has been provided. "
                "Enter a 1d out-of-transit stellar spectrum from eg. HARPS."
            )
        F_obsc = compute_obscured_flux(res, D, F_oot_HARPS)
        F_obsc_shifted = inverse_shift(F_obsc, in_transit_idx, wl, xp, vsini)

    else:
        raise Exception(
            "Invalid instrument has been entered. Select either "
            "HARPS, StarRotator or MAROON-X."
        )

    FstarN = weighted_average(
        F_obsc_shifted, r[in_transit_idx.flatten()], Rp_Rs, u1, u2
    )

    return FstarN
