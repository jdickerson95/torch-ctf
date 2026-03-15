"""CTF with sample thickness: amplitude transfer vs power spectrum (Thon rings)."""

from __future__ import annotations

from typing import Any, Literal

import einops
import torch
from torch_grid_utils.fftfreq_grid import fftfreq_grid, transform_fftfreq_grid

from torch_ctf.ctf_1d import _setup_ctf_1d
from torch_ctf.ctf_2d import _setup_ctf_2d
from torch_ctf.ctf_aberrations import (
    apply_even_zernikes,
    apply_odd_zernikes,
    calculate_relativistic_electron_wavelength,
)
from torch_ctf.ctf_lpp import calc_LPP_phase
from torch_ctf.ctf_utils import calculate_total_phase_shift


def _sinc_sin_over_x(x: torch.Tensor) -> torch.Tensor:
    """Compute sin(x)/x with numerical limit 1 at x=0.

    Parameters
    ----------
    x : torch.Tensor
        Argument in radians; any shape.

    Returns
    -------
    torch.Tensor
        Same shape as ``x``.
    """
    return torch.sinc(x / torch.pi)


def _ctf_from_thickness(
    return_power_spectrum: bool,
    lambda_angstrom: torch.Tensor,
    g_squared_angstrom_inv2: torch.Tensor,
    chi_radians: torch.Tensor,
    thickness_angstrom: torch.Tensor,
) -> torch.Tensor:
    """Apply sample-thickness modulation at each spatial frequency (internal).

    Parameters
    ----------
    return_power_spectrum : bool
        If False, amplitude CTF proportional to
        sin(pi*lambda*|g|^2*t/2)/(...)*sin(chi). If True, expected power-spectrum
        form 0.5*(1 - sin(pi*lambda*|g|^2*t)/(...)*cos(2*chi)).
    lambda_angstrom : torch.Tensor
        Electron wavelength in Angstroms; broadcastable to ``g_squared_angstrom_inv2``.
    g_squared_angstrom_inv2 : torch.Tensor
        Squared spatial frequency |g|^2 in Angstroms^-2.
    chi_radians : torch.Tensor
        Aberration phase chi in radians, same broadcast shape as effective CTF grid.
    thickness_angstrom : float | torch.Tensor
        Sample thickness t in Angstroms.

    Returns
    -------
    torch.Tensor
        Real-valued thickness-modulated CTF factor on the frequency grid.
    """
    t = torch.as_tensor(
        thickness_angstrom, dtype=lambda_angstrom.dtype, device=lambda_angstrom.device
    )
    # Broadcast lambda, t to g^2 / chi
    while lambda_angstrom.ndim < g_squared_angstrom_inv2.ndim:
        lambda_angstrom = lambda_angstrom.unsqueeze(-1)
    while t.ndim < g_squared_angstrom_inv2.ndim:
        t = t.unsqueeze(-1)

    if not return_power_spectrum:
        z = torch.pi * lambda_angstrom * g_squared_angstrom_inv2 * t / 2
        return _sinc_sin_over_x(z) * torch.sin(chi_radians)
    xi = torch.pi * lambda_angstrom * g_squared_angstrom_inv2 * t
    return 0.5 * (1.0 - _sinc_sin_over_x(xi) * torch.cos(2.0 * chi_radians))


def calculate_ctf_thickness_1d(
    return_power_spectrum: bool,
    sample_thickness_angstrom: float | torch.Tensor,
    defocus: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    phase_shift: float | torch.Tensor,
    pixel_size: float | torch.Tensor,
    n_samples: int,
    oversampling_factor: int,
) -> torch.Tensor:
    """1D CTF including sample thickness (amplitude or Thon-ring power spectrum).

    Parameters
    ----------
    return_power_spectrum : bool
        If False, amplitude-transfer CTF with thickness. If True, expected
        power-spectrum (Thon ring) form.
    sample_thickness_angstrom : float | torch.Tensor
        Sample thickness in Angstroms.
    defocus : float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
    voltage : float | torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration : float | torch.Tensor
        Spherical aberration in millimeters (mm).
    amplitude_contrast : float | torch.Tensor
        Fraction of amplitude contrast (value in range [0, 1]).
    phase_shift : float | torch.Tensor
        Angle of phase shift applied to CTF in degrees.
    pixel_size : float | torch.Tensor
        Pixel size in Angstroms per pixel (Angstrom px^-1).
    n_samples : int
        Number of samples along the 1D frequency axis.
    oversampling_factor : int
        Factor by which to oversample before averaging back to ``n_samples``.

    Returns
    -------
    ctf : torch.Tensor
        Thickness-modulated 1D CTF (real-valued).

    """
    (
        defocus,
        voltage,
        spherical_aberration,
        amplitude_contrast,
        phase_shift,
        g2,
        oversampling_factor_out,
    ) = _setup_ctf_1d(
        defocus=defocus,
        voltage=voltage,
        spherical_aberration=spherical_aberration,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift,
        pixel_size=pixel_size,
        n_samples=n_samples,
        oversampling_factor=oversampling_factor,
    )

    chi = calculate_total_phase_shift(
        defocus_um=defocus,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        phase_shift_degrees=phase_shift,
        amplitude_contrast_fraction=amplitude_contrast,
        fftfreq_grid_angstrom_squared=g2,
    )
    lam = calculate_relativistic_electron_wavelength(voltage * 1e3) * 1e10
    ctf = _ctf_from_thickness(
        return_power_spectrum, lam, g2, chi, sample_thickness_angstrom
    )

    if oversampling_factor_out > 1:
        ctf = einops.reduce(ctf, "... os k -> ... k", reduction="mean")
    return ctf


def calculate_ctf_thickness_2d(
    return_power_spectrum: bool,
    sample_thickness_angstrom: float | torch.Tensor,
    defocus: float | torch.Tensor,
    astigmatism: float | torch.Tensor,
    astigmatism_angle: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    phase_shift: float | torch.Tensor,
    pixel_size: float | torch.Tensor,
    image_shape: tuple[int, int],
    rfft: bool,
    fftshift: bool,
    beam_tilt_mrad: torch.Tensor | None = None,
    even_zernike_coeffs: dict | None = None,
    odd_zernike_coeffs: dict | None = None,
    transform_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D CTF including sample thickness (amplitude or Thon-ring power spectrum).

    NOTE: The device of the input tensors is inferred from the ``defocus`` tensor.

    Odd Zernikes and beam tilt are applied only when ``return_power_spectrum`` is
    False; they are ignored for the power-spectrum form.

    Parameters
    ----------
    return_power_spectrum : bool
        If False, amplitude CTF with thickness (may be complex if beam tilt or odd
        Zernikes are set). If True, real power-spectrum form (Thon rings).
    sample_thickness_angstrom : float | torch.Tensor
        Sample thickness in Angstroms.
    defocus : float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
        ``(defocus_u + defocus_v) / 2``
    astigmatism : float | torch.Tensor
        Amount of astigmatism in micrometers.
        ``(defocus_u - defocus_v) / 2``
    astigmatism_angle : float | torch.Tensor
        Angle of astigmatism in degrees. 0 places ``defocus_u`` along the y-axis.
    voltage : float | torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration : float | torch.Tensor
        Spherical aberration in millimeters (mm).
    amplitude_contrast : float | torch.Tensor
        Fraction of amplitude contrast (value in range [0, 1]).
    phase_shift : float | torch.Tensor
        Angle of phase shift applied to CTF in degrees.
    pixel_size : float | torch.Tensor
        Pixel size in Angstroms per pixel (Angstrom px^-1).
    image_shape : tuple[int, int]
        Shape ``(H, W)`` of 2D images onto which the CTF is applied.
    rfft : bool
        If True, CTF matches the non-redundant half-grid of an ``rfft``.
    fftshift : bool
        Whether frequency grids use fftshift layout.
    beam_tilt_mrad : torch.Tensor | None
        Beam tilt in milliradians ``[bx, by]``. Only used if
        ``return_power_spectrum`` is False.
    even_zernike_coeffs : dict | None
        Even Zernike coefficients.
        Example: ``{"Z44c": 0.1, "Z44s": 0.2, "Z60": 0.3}``
    odd_zernike_coeffs : dict | None
        Odd Zernike coefficients. Only used if ``return_power_spectrum`` is False.
        Example: ``{"Z31c": 0.1, "Z31s": 0.2, "Z33c": 0.3, "Z33s": 0.4}``
    transform_matrix : torch.Tensor | None
        Optional 2x2 real-space transformation matrix ``A``; frequency grid uses
        ``(A^-1)^T``.

    Returns
    -------
    ctf : torch.Tensor
        Thickness-modulated 2D CTF. Real if ``return_power_spectrum`` or no odd
        aberrations; otherwise complex.

    """
    (
        defocus,
        voltage,
        spherical_aberration,
        amplitude_contrast,
        phase_shift,
        g2,
        rho,
        theta,
    ) = _setup_ctf_2d(
        defocus=defocus,
        astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle,
        voltage=voltage,
        spherical_aberration=spherical_aberration,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift,
        pixel_size=pixel_size,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        transform_matrix=transform_matrix,
    )

    chi = calculate_total_phase_shift(
        defocus_um=defocus,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        phase_shift_degrees=phase_shift,
        amplitude_contrast_fraction=amplitude_contrast,
        fftfreq_grid_angstrom_squared=g2,
    )
    if even_zernike_coeffs is not None:
        chi = apply_even_zernikes(even_zernike_coeffs, chi, rho, theta)

    lam = calculate_relativistic_electron_wavelength(voltage * 1e3) * 1e10
    ctf = _ctf_from_thickness(
        return_power_spectrum, lam, g2, chi, sample_thickness_angstrom
    )

    if return_power_spectrum:
        return ctf
    if odd_zernike_coeffs is None and beam_tilt_mrad is None:
        return ctf

    antisymmetric_phase_shift = apply_odd_zernikes(
        odd_zernikes=odd_zernike_coeffs,
        rho=rho,
        theta=theta,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        beam_tilt_mrad=beam_tilt_mrad,
    )
    return ctf * torch.exp(1j * antisymmetric_phase_shift)


def calculate_ctf_thickness_lpp(
    return_power_spectrum: bool,
    sample_thickness_angstrom: float | torch.Tensor,
    defocus: float | torch.Tensor,
    astigmatism: float | torch.Tensor,
    astigmatism_angle: float | torch.Tensor,
    voltage: float | torch.Tensor,
    spherical_aberration: float | torch.Tensor,
    amplitude_contrast: float | torch.Tensor,
    pixel_size: float | torch.Tensor,
    image_shape: tuple[int, int],
    rfft: bool,
    fftshift: bool,
    NA: float,
    laser_wavelength_angstrom: float,
    focal_length_angstrom: float,
    laser_xy_angle_deg: float,
    laser_xz_angle_deg: float,
    laser_long_offset_angstrom: float,
    laser_trans_offset_angstrom: float,
    laser_polarization_angle_deg: float,
    peak_phase_deg: float,
    dual_laser: bool = False,
    beam_tilt_mrad: torch.Tensor | None = None,
    even_zernike_coeffs: dict | None = None,
    odd_zernike_coeffs: dict | None = None,
    transform_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """Laser phase plate (LPP) 2D CTF with sample thickness.

    Same geometry and laser arguments as :func:`calc_LPP_ctf_2D`, plus thickness.
    NOTE: The device of the input tensors is inferred from the ``defocus`` tensor.

    Odd Zernikes and beam tilt apply only when ``return_power_spectrum`` is False.

    Parameters
    ----------
    return_power_spectrum : bool
        If False, amplitude CTF with thickness. If True, Thon-ring power-spectrum form
        (real; beam tilt and odd Zernikes ignored).
    sample_thickness_angstrom : float | torch.Tensor
        Sample thickness in Angstroms.
    defocus : float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
        ``(defocus_u + defocus_v) / 2``
    astigmatism : float | torch.Tensor
        Astigmatism in micrometers. ``(defocus_u - defocus_v) / 2``
    astigmatism_angle : float | torch.Tensor
        Astigmatism angle in degrees.
    voltage : float | torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration : float | torch.Tensor
        Spherical aberration in millimeters (mm).
    amplitude_contrast : float | torch.Tensor
        Fraction of amplitude contrast (value in range [0, 1]).
    pixel_size : float | torch.Tensor
        Pixel size in Angstroms per pixel (Angstrom px^-1).
    image_shape : tuple[int, int]
        Shape ``(H, W)`` of 2D images.
    rfft : bool
        If True, CTF for ``rfft`` half-grid.
    fftshift : bool
        fftshift layout for frequency grids.
    NA : float
        Numerical aperture of the laser.
    laser_wavelength_angstrom : float
        Laser wavelength in Angstroms.
    focal_length_angstrom : float
        Focal length in Angstroms.
    laser_xy_angle_deg : float
        Laser rotation in the xy plane (degrees).
    laser_xz_angle_deg : float
        Laser angle in the xz plane (degrees).
    laser_long_offset_angstrom : float
        Longitudinal laser offset in Angstroms.
    laser_trans_offset_angstrom : float
        Transverse laser offset in Angstroms.
    laser_polarization_angle_deg : float
        Laser polarization angle in degrees.
    peak_phase_deg : float
        Desired peak laser phase in degrees.
    dual_laser : bool, optional
        If True, add a second perpendicular laser in xy. Default False.
    beam_tilt_mrad : torch.Tensor | None
        Beam tilt ``[bx, by]`` in mrad. Only if ``return_power_spectrum`` is False.
    even_zernike_coeffs : dict | None
        Even Zernike coefficients.
    odd_zernike_coeffs : dict | None
        Odd Zernike coefficients. Only if ``return_power_spectrum`` is False.
    transform_matrix : torch.Tensor | None
        Optional 2x2 real-space matrix ``A`` for anisotropic magnification.

    Returns
    -------
    ctf : torch.Tensor
        Thickness-modulated LPP CTF (real for power spectrum or no odd phase).

    """
    (
        defocus,
        voltage,
        spherical_aberration,
        amplitude_contrast,
        _,
        g2,
        rho,
        theta,
    ) = _setup_ctf_2d(
        defocus=defocus,
        astigmatism=astigmatism,
        astigmatism_angle=astigmatism_angle,
        voltage=voltage,
        spherical_aberration=spherical_aberration,
        amplitude_contrast=amplitude_contrast,
        phase_shift=torch.tensor(0.0),
        pixel_size=pixel_size,
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        transform_matrix=transform_matrix,
    )

    if isinstance(defocus, torch.Tensor):
        device = defocus.device
    else:
        device = torch.device("cpu")

    pixel_size_tensor = torch.as_tensor(pixel_size, dtype=torch.float, device=device)
    image_shape_tensor = torch.as_tensor(image_shape, dtype=torch.int, device=device)

    fft_freq_grid = fftfreq_grid(
        image_shape=image_shape_tensor,
        rfft=rfft,
        fftshift=fftshift,
        norm=False,
        device=device,
    )
    if transform_matrix is not None:
        fft_freq_grid = transform_fftfreq_grid(
            frequency_grid=fft_freq_grid,
            real_space_matrix=transform_matrix,
            device=device,
        )
    fft_freq_grid = fft_freq_grid / einops.rearrange(
        pixel_size_tensor, "... -> ... 1 1 1"
    )

    if dual_laser:
        phase1 = calc_LPP_phase(
            fft_freq_grid=fft_freq_grid,
            NA=NA,
            laser_wavelength_angstrom=laser_wavelength_angstrom,
            focal_length_angstrom=focal_length_angstrom,
            laser_xy_angle_deg=laser_xy_angle_deg,
            laser_xz_angle_deg=laser_xz_angle_deg,
            laser_long_offset_angstrom=laser_long_offset_angstrom,
            laser_trans_offset_angstrom=laser_trans_offset_angstrom,
            laser_polarization_angle_deg=laser_polarization_angle_deg,
            peak_phase_deg=peak_phase_deg,
            voltage=voltage,
        )
        phase2 = calc_LPP_phase(
            fft_freq_grid=fft_freq_grid,
            NA=NA,
            laser_wavelength_angstrom=laser_wavelength_angstrom,
            focal_length_angstrom=focal_length_angstrom,
            laser_xy_angle_deg=laser_xy_angle_deg + 90,
            laser_xz_angle_deg=laser_xz_angle_deg,
            laser_long_offset_angstrom=laser_long_offset_angstrom,
            laser_trans_offset_angstrom=laser_trans_offset_angstrom,
            laser_polarization_angle_deg=laser_polarization_angle_deg,
            peak_phase_deg=peak_phase_deg,
            voltage=voltage,
        )
        laser_phase_radians = phase1 + phase2
    else:
        laser_phase_radians = calc_LPP_phase(
            fft_freq_grid=fft_freq_grid,
            NA=NA,
            laser_wavelength_angstrom=laser_wavelength_angstrom,
            focal_length_angstrom=focal_length_angstrom,
            laser_xy_angle_deg=laser_xy_angle_deg,
            laser_xz_angle_deg=laser_xz_angle_deg,
            laser_long_offset_angstrom=laser_long_offset_angstrom,
            laser_trans_offset_angstrom=laser_trans_offset_angstrom,
            laser_polarization_angle_deg=laser_polarization_angle_deg,
            peak_phase_deg=peak_phase_deg,
            voltage=voltage,
        )

    laser_phase_degrees = torch.rad2deg(laser_phase_radians)
    chi = calculate_total_phase_shift(
        defocus_um=defocus,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        phase_shift_degrees=laser_phase_degrees,
        amplitude_contrast_fraction=amplitude_contrast,
        fftfreq_grid_angstrom_squared=g2,
    )
    if even_zernike_coeffs is not None:
        chi = apply_even_zernikes(even_zernike_coeffs, chi, rho, theta)

    lam = calculate_relativistic_electron_wavelength(voltage * 1e3) * 1e10
    ctf = _ctf_from_thickness(
        return_power_spectrum, lam, g2, chi, sample_thickness_angstrom
    )

    if return_power_spectrum:
        return ctf
    if odd_zernike_coeffs is None and beam_tilt_mrad is None:
        return ctf

    antisymmetric_phase_shift = apply_odd_zernikes(
        odd_zernikes=odd_zernike_coeffs,
        rho=rho,
        theta=theta,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        beam_tilt_mrad=beam_tilt_mrad,
    )
    return ctf * torch.exp(1j * antisymmetric_phase_shift)


def calculate_ctf_with_thickness(
    geometry: Literal["1d", "2d", "lpp"],
    return_power_spectrum: bool,
    sample_thickness_angstrom: float | torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """Dispatch to :func:`calculate_ctf_thickness_1d`, ``_2d``, or ``_lpp``.

    Parameters
    ----------
    geometry : Literal["1d", "2d", "lpp"]
        Which CTF API to use.
    return_power_spectrum : bool
        Passed through; if True, Thon-ring form (2D/LPP ignore odd Zernikes and
        beam tilt).
    sample_thickness_angstrom : float | torch.Tensor
        Sample thickness in Angstroms.
    **kwargs
        Arguments for the selected function (same as ``calculate_ctf_1d`` /
        ``calculate_ctf_2d`` / ``calc_LPP_ctf_2D`` for the corresponding geometry,
        excluding the thickness-specific leading parameters).

    Returns
    -------
    ctf : torch.Tensor
        Output of the dispatched thickness CTF function.

    Raises
    ------
    ValueError
        If ``geometry`` is not ``"1d"``, ``"2d"``, or ``"lpp"``.

    """
    if geometry not in ("1d", "2d", "lpp"):
        raise ValueError(f"geometry must be '1d', '2d', or 'lpp', got {geometry!r}")
    if geometry == "1d":
        return calculate_ctf_thickness_1d(
            return_power_spectrum,
            sample_thickness_angstrom,
            **kwargs,
        )
    if geometry == "2d":
        return calculate_ctf_thickness_2d(
            return_power_spectrum,
            sample_thickness_angstrom,
            **kwargs,
        )
    return calculate_ctf_thickness_lpp(
        return_power_spectrum,
        sample_thickness_angstrom,
        **kwargs,
    )
