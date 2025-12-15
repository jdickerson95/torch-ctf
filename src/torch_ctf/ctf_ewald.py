"""CTF calculation with Ewald sphere correction."""

import torch

from torch_ctf.ctf_2d import _setup_ctf_2d
from torch_ctf.ctf_aberrations import apply_even_zernikes, apply_odd_zernikes
from torch_ctf.ctf_utils import calculate_total_phase_shift


def calculate_ctfp_and_ctfq_2d(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate CTFP and CTFQ for a 2D image.

    Parameters
    ----------
    defocus : float | torch.Tensor
        Defocus in micrometers, positive is underfocused.
        `(defocus_u + defocus_v) / 2`
    astigmatism : float | torch.Tensor
        Amount of astigmatism in micrometers.
        `(defocus_u - defocus_v) / 2`
    astigmatism_angle : float | torch.Tensor
        Angle of astigmatism in degrees. 0 places `defocus_u` along the y-axis.
    voltage : float | torch.Tensor
        Acceleration voltage in kilovolts (kV).
    spherical_aberration : float | torch.Tensor
        Spherical aberration in millimeters (mm).
    amplitude_contrast : float | torch.Tensor
        Fraction of amplitude contrast (value in range [0, 1]).
    phase_shift : float | torch.Tensor
        Angle of phase shift applied to CTF in degrees.
    pixel_size : float | torch.Tensor
        Pixel size in Angströms per pixel (Å px⁻¹).
    image_shape : tuple[int, int]
        Shape of 2D images onto which CTF will be applied.
    rfft : bool
        Generate the CTF containing only the non-redundant half transform from a rfft.
    fftshift : bool
        Whether to apply fftshift on the resulting CTF images.
    beam_tilt_mrad : torch.Tensor | None
        Beam tilt in milliradians. [bx, by] in mrad
    even_zernike_coeffs : dict | None
        Even Zernike coefficients.
        Example: {"Z44c": 0.1, "Z44s": 0.2, "Z60": 0.3}
    odd_zernike_coeffs : dict | None
        Odd Zernike coefficients.
        Example: {"Z31c": 0.1, "Z31s": 0.2, "Z33c": 0.3, "Z33s": 0.4}

    Returns
    -------
    ctfp : torch.Tensor
        The P component of the CTF for the given parameters (complex tensor).
    ctfq : torch.Tensor
        The Q component of the CTF for the given parameters (complex tensor).
    """
    (
        defocus,
        voltage,
        spherical_aberration,
        amplitude_contrast,
        phase_shift,
        fft_freq_grid_squared,
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
    )

    total_phase_shift = calculate_total_phase_shift(
        defocus_um=defocus,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        phase_shift_degrees=phase_shift,
        amplitude_contrast_fraction=amplitude_contrast,
        fftfreq_grid_angstrom_squared=fft_freq_grid_squared,
    )

    if even_zernike_coeffs is not None:
        total_phase_shift = apply_even_zernikes(
            even_zernike_coeffs,
            total_phase_shift,
            rho,
            theta,
        )

    # calculate ctf
    # ctfp: real = -sin, imag = +cos
    ctfp = torch.complex(
        -torch.sin(total_phase_shift),
        torch.cos(total_phase_shift),
    )

    # ctfq: real = -sin, imag = -cos
    ctfq = torch.complex(
        -torch.sin(total_phase_shift),
        -torch.cos(total_phase_shift),
    )

    if odd_zernike_coeffs is None:
        return ctfp, ctfq

    antisymmetric_phase_shift = apply_odd_zernikes(
        odd_zernikes=odd_zernike_coeffs,
        rho=rho,
        theta=theta,
        voltage_kv=voltage,
        spherical_aberration_mm=spherical_aberration,
        beam_tilt_mrad=beam_tilt_mrad,
    )
    return ctfp * torch.exp(1j * antisymmetric_phase_shift), ctfq * torch.exp(
        1j * antisymmetric_phase_shift
    )
