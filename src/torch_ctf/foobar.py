from pathlib import Path
import ast

src_path = Path('tests/test_torch_ctf.py')
source = src_path.read_text()
module = ast.parse(source)
lines = source.splitlines()

def seg(node):
    return '\n'.join(lines[node.lineno-1:node.end_lineno]) + '\n'

func_nodes = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')}
expected_node = next(n for n in module.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='EXPECTED_2D' for t in n.targets))
expected_src = seg(expected_node)

groups = {
    'tests/test_ctf_basics.py': [
        'test_1d_ctf_single',
        'test_1d_ctf_batch',
        'test_calculate_relativistic_electron_wavelength',
        'test_calculate_relativistic_electron_wavelength_tensor',
        'test_2d_ctf_batch',
        'test_2d_ctf_astigmatism',
        'test_2d_ctf_rfft',
        'test_2d_ctf_fftshift',
        'test_2d_ctf_transform_matrix',
        'test_calculate_defocus_phase_aberration',
        'test_calculate_additional_phase_shift',
        'test_calculate_amplitude_contrast_equivalent_phase_shift',
        'test_calculate_total_phase_shift',
        'test_2d_ctf_with_zernikes',
        'test_2d_ctf_with_beam_tilt_only',
        'test_calculate_ctf_2d_return_complex_ctf_symmetric_path',
        'test_calculate_ctf_2d_return_complex_ctf_rfft_shape_and_unit_modulus',
        'test_calculate_ctf_2d_return_complex_ctf_beam_tilt_matches_manual_phase',
    ],
    'tests/test_ctf_ewald.py': [
        'test_calculate_ctfp_and_ctfq_2d',
        'test_calculate_ctfp_and_ctfq_2d_discontinuity_angle_zero',
        'test_calculate_ctfp_and_ctfq_2d_discontinuity_angle_22_5',
        'test_calculate_ctfp_and_ctfq_2d_discontinuity_angle_with_blur',
        'test_get_ctf_weighting',
    ],
    'tests/test_ctf_aberrations.py': [
        'test_beam_tilt_to_zernike_coeffs',
        'test_resolve_odd_zernikes_beam_tilt_only',
        'test_resolve_odd_zernikes_zernike_only',
        'test_resolve_odd_zernikes_both',
        'test_resolve_odd_zernikes_none',
        'test_apply_odd_zernikes',
        'test_apply_odd_zernikes_trefoil',
        'test_apply_odd_zernikes_invalid',
        'test_apply_even_zernikes',
        'test_apply_even_zernikes_invalid',
        'test_apply_even_zernikes_tensor_coeffs',
        'test_apply_even_zernikes_mixed_coeffs',
        'test_apply_even_zernikes_type_error',
        'test_apply_odd_zernikes_none',
        'test_apply_odd_zernikes_float_coeffs',
        'test_apply_odd_zernikes_mixed_coeffs',
        'test_apply_odd_zernikes_with_beam_tilt',
        'test_apply_odd_zernikes_type_error',
        'test_resolve_odd_zernikes_with_trefoil',
        'test_resolve_odd_zernikes_beam_tilt_with_trefoil',
        'test_beam_tilt_to_zernike_coeffs_broadcasting',
    ],
    'tests/test_ctf_lpp.py': [
        'test_calculate_relativistic_gamma',
        'test_calculate_relativistic_beta',
        'test_initialize_laser_params',
        'test_make_laser_coords',
        'test_get_eta',
        'test_get_eta0_from_peak_phase_deg',
        'test_calc_LPP_phase',
        'test_calc_LPP_ctf_2D',
        'test_calc_LPP_ctf_2D_dual_laser',
        'test_calc_LPP_ctf_2D_with_zernikes',
        'test_calc_LPP_ctf_2D_with_beam_tilt_only',
        'test_calc_LPP_ctf_2D_return_complex_ctf_symmetric_path',
        'test_calc_LPP_ctf_2D_return_complex_ctf_beam_tilt_unit_modulus',
        'test_calc_LPP_ctf_2D_return_complex_ctf_dual_laser',
        'test_calc_LPP_ctf_2D_with_even_zernikes',
        'test_calc_LPP_ctf_2D_with_transform_matrix',
    ],
    'tests/test_ctf_thickness.py': [
        'test_ctf_thickness_1d_amplitude_small_t_matches_sin_chi',
        'test_ctf_thickness_1d_power_spectrum_small_t_matches_sin_squared_chi',
        'test_ctf_thickness_router_matches_explicit_1d',
        'test_ctf_thickness_2d_amplitude_small_t_matches_sin_chi',
        'test_ctf_thickness_lpp_shape_and_formulations_differ',
        'test_ctf_thickness_2d_power_spectrum_ignores_beam_tilt',
        'test_ctf_thickness_invalid_geometry',
        'test_ctf_from_thickness_broadcast_unsqueeze',
        'test_ctf_thickness_2d_with_even_zernikes',
        'test_ctf_thickness_2d_amplitude_with_beam_tilt_returns_complex',
        'test_ctf_thickness_router_2d_matches_explicit',
        'test_ctf_thickness_router_lpp_matches_explicit',
        'test_ctf_thickness_lpp_defocus_float_device_cpu',
        'test_ctf_thickness_lpp_with_transform_matrix',
        'test_ctf_thickness_lpp_dual_laser',
        'test_ctf_thickness_lpp_with_even_zernikes',
        'test_ctf_thickness_lpp_amplitude_with_beam_tilt_returns_complex',
    ],
}

imports = {
    'tests/test_ctf_basics.py': '''"""Tests for baseline 1D/2D CTF and phase utilities."""\n\nimport torch\n\nfrom torch_ctf import (\n    apply_odd_zernikes,\n    calculate_additional_phase_shift,\n    calculate_amplitude_contrast_equivalent_phase_shift,\n    calculate_ctf_1d,\n    calculate_ctf_2d,\n    calculate_defocus_phase_aberration,\n    calculate_relativistic_electron_wavelength,\n    calculate_total_phase_shift,\n)\n''',
    'tests/test_ctf_ewald.py': '''"""Tests for Ewald-sphere CTF outputs and weighting."""\n\nimport torch\n\nfrom torch_ctf import calculate_ctfp_and_ctfq_2d, get_ctf_weighting\n''',
    'tests/test_ctf_aberrations.py': '''"""Tests for aberration helpers and Zernike handling."""\n\nimport pytest\nimport torch\n\nfrom torch_ctf import (\n    apply_even_zernikes,\n    apply_odd_zernikes,\n    beam_tilt_to_zernike_coeffs,\n    resolve_odd_zernikes,\n)\n''',
    'tests/test_ctf_lpp.py': '''"""Tests for laser phase plate utilities and CTF."""\n\nimport torch\n\nfrom torch_ctf import (\n    calc_LPP_ctf_2D,\n    calc_LPP_phase,\n    calculate_relativistic_beta,\n    calculate_relativistic_gamma,\n    get_eta,\n    get_eta0_from_peak_phase_deg,\n    initialize_laser_params,\n    make_laser_coords,\n)\n''',
    'tests/test_ctf_thickness.py': '''"""Tests for sample-thickness CTF variants."""\n\nimport pytest\nimport torch\n\nfrom torch_ctf.ctf_thickness import (\n    _ctf_from_thickness,\n    calculate_ctf_thickness_1d,\n    calculate_ctf_thickness_2d,\n    calculate_ctf_thickness_lpp,\n    calculate_ctf_with_thickness,\n)\n''',
}

for path, names in groups.items():
    content = [imports[path], '\n']
    if path == 'tests/test_ctf_basics.py':
        content.append(expected_src)
        content.append('\n')
    for name in names:
        content.append(seg(func_nodes[name]))
        content.append('\n')
    Path(path).write_text(''.join(content).rstrip() + '\n')

src_path.unlink()
print('split complete')
