"""Build a citation-anchored post-1900 formula-level transfer benchmark.

This script performs no model fitting or representation training.  It evaluates
curated one-dimensional reductions of laws first published after 1900 on fixed,
physically meaningful reduced-variable ranges.  Multiple records per law vary
sampling, nuisance parameters and measurement noise; the scientific inference
unit remains the source-law family.

Planck's 1900 law is deliberately excluded by the strict ``first_valid_year >
1900`` rule.  The benchmark is a formula-level temporal-transfer test, not a
real-measurement benchmark and not a claim that publication year is identifiable
from curve shape.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SEED = 20260906
REPEATS = 8
POINTS = 120


# Each row identifies an independent named source-law family.  ``equation`` is
# the nondimensional one-dimensional reduction actually sampled below, not an
# assertion that every physical degree of freedom has been retained.
REGISTRY = [
    dict(family_id="richardson_dushman_emission", law_name="Richardson thermionic-emission law",
         first_valid_year=1901, domain="statistical_physics", generator="richardson_dushman",
         equation="j(t) = t^2 exp(-a/t)", x_range=[0.20, 2.00], sampling="linear",
         shape_class="smooth_composite", signature="power_times_inverse_temperature_exponential",
         citation="O. W. Richardson, On the negative radiation from hot platinum, Proc. Cambridge Philos. Soc. 11, 286-295 (1901)."),
    dict(family_id="radioactive_decay", law_name="Rutherford-Soddy radioactive-decay law",
         first_valid_year=1902, domain="nuclear_physics", generator="radioactive_decay",
         equation="n(t) = exp(-lambda t)", x_range=[0.0, 6.0], sampling="linear",
         shape_class="declared_dictionary_overlap", signature="exponential_decay",
         citation="E. Rutherford and F. Soddy, The cause and nature of radioactivity, Philos. Mag. 4, 370-396 (1902)."),
    dict(family_id="special_relativity", law_name="Lorentz factor in special relativity",
         first_valid_year=1905, domain="relativity", generator="lorentz_gamma",
         equation="gamma(beta) = (1-beta^2)^(-1/2)", x_range=[0.02, 0.98], sampling="linear",
         shape_class="smooth_composite", signature="relativistic_divergence",
         citation="A. Einstein, Zur Elektrodynamik bewegter Korper, Ann. Phys. 322, 891-921 (1905), doi:10.1002/andp.19053221004."),
    dict(family_id="photoelectric_threshold", law_name="Einstein photoelectric relation",
         first_valid_year=1905, domain="quantum_physics", generator="photoelectric_threshold",
         equation="k(x) = max(x-x0, 0)", x_range=[0.20, 3.20], sampling="linear",
         shape_class="threshold_or_piecewise", signature="thresholded_affine",
         citation="A. Einstein, Uber einen die Erzeugung und Verwandlung des Lichtes betreffenden heuristischen Gesichtspunkt, Ann. Phys. 322, 132-148 (1905), doi:10.1002/andp.19053220607."),
    dict(family_id="brownian_diffusion", law_name="Einstein mean-square displacement",
         first_valid_year=1905, domain="statistical_physics", generator="brownian_msd",
         equation="msd(t) = a t", x_range=[0.05, 6.0], sampling="linear",
         shape_class="declared_dictionary_overlap", signature="linear",
         citation="A. Einstein, Uber die von der molekularkinetischen Theorie der Warme geforderte Bewegung von in ruhenden Flussigkeiten suspendierten Teilchen, Ann. Phys. 322, 549-560 (1905), doi:10.1002/andp.19053220806."),
    dict(family_id="langevin_paramagnetism", law_name="Langevin paramagnetism function",
         first_valid_year=1905, domain="statistical_physics", generator="langevin_function",
         equation="m(x) = coth(x)-1/x", x_range=[0.05, 8.0], sampling="linear",
         shape_class="smooth_composite", signature="coth_minus_inverse",
         citation="P. Langevin, Magnetisme et theorie des electrons, Ann. Chim. Phys. 5, 70-127 (1905)."),
    dict(family_id="einstein_solid", law_name="Einstein solid heat capacity",
         first_valid_year=1907, domain="condensed_matter", generator="einstein_heat_capacity",
         equation="c(x) = x^2 exp(x)/(exp(x)-1)^2", x_range=[0.12, 10.0], sampling="log",
         shape_class="smooth_composite", signature="einstein_oscillator_heat_capacity",
         citation="A. Einstein, Die Plancksche Theorie der Strahlung und die Theorie der spezifischen Warme, Ann. Phys. 327, 180-190 (1907), doi:10.1002/andp.19063270110."),
    dict(family_id="curie_weiss_susceptibility", law_name="Curie-Weiss susceptibility",
         first_valid_year=1907, domain="condensed_matter", generator="curie_weiss",
         equation="chi(t) = 1/(t-1)", x_range=[1.04, 5.0], sampling="log",
         shape_class="smooth_composite", signature="shifted_inverse",
         citation="P. Weiss, L'hypothese du champ moleculaire et la propriete ferromagnetique, J. Phys. Theor. Appl. 6, 661-690 (1907), doi:10.1051/jphystap:019070060066100."),
    dict(family_id="hill_binding", law_name="Hill cooperative-binding equation",
         first_valid_year=1910, domain="biophysics", generator="hill_binding",
         equation="theta(x) = x^n/(1+x^n)", x_range=[0.04, 8.0], sampling="log",
         shape_class="smooth_composite", signature="cooperative_sigmoid",
         citation="A. V. Hill, The possible effects of the aggregation of the molecules of haemoglobin on its dissociation curves, J. Physiol. 40, iv-vii (1910)."),
    dict(family_id="geiger_nuttall", law_name="Geiger-Nuttall alpha-decay relation",
         first_valid_year=1911, domain="nuclear_physics", generator="geiger_nuttall",
         equation="lambda(e) = exp(a-b/sqrt(e))", x_range=[0.20, 5.0], sampling="linear",
         shape_class="smooth_composite", signature="inverse_root_exponential",
         citation="H. Geiger and J. M. Nuttall, The ranges of the alpha particles from various radioactive substances and a relation between range and period of transformation, Philos. Mag. 22, 613-621 (1911)."),
    dict(family_id="rutherford_scattering", law_name="Rutherford scattering angular law",
         first_valid_year=1911, domain="nuclear_physics", generator="rutherford_scattering",
         equation="i(theta) = sin(theta/2)^(-4)", x_range=[0.22, 2.90], sampling="linear",
         shape_class="smooth_composite", signature="inverse_trigonometric_power",
         citation="E. Rutherford, The scattering of alpha and beta particles by matter and the structure of the atom, Philos. Mag. 21, 669-688 (1911), doi:10.1080/14786440508637080."),
    dict(family_id="child_langmuir_space_charge", law_name="Child-Langmuir space-charge law",
         first_valid_year=1911, domain="electromagnetism", generator="child_langmuir",
         equation="j(v) = a v^(3/2)", x_range=[0.05, 5.0], sampling="linear",
         shape_class="declared_dictionary_overlap", signature="power_3_over_2",
         citation="C. D. Child, Discharge from hot CaO, Phys. Rev. (Series I) 32, 492-511 (1911), doi:10.1103/PhysRevSeriesI.32.492."),
    dict(family_id="debye_solid", law_name="Debye heat-capacity law",
         first_valid_year=1912, domain="condensed_matter", generator="debye_heat_capacity",
         equation="c(t) = 9 t^3 integral_0^(1/t) u^4 exp(u)/(exp(u)-1)^2 du", x_range=[0.04, 2.5], sampling="log",
         shape_class="special_function_or_integral", signature="debye_integral",
         citation="P. Debye, Zur Theorie der spezifischen Warmen, Ann. Phys. 344, 789-839 (1912), doi:10.1002/andp.19123441404."),
    dict(family_id="bragg_diffraction", law_name="Bragg diffraction law",
         first_valid_year=1913, domain="quantum_and_atomic", generator="bragg_diffraction",
         equation="theta(x) = arcsin(x)", x_range=[0.03, 0.97], sampling="linear",
         shape_class="smooth_composite", signature="inverse_trigonometric",
         citation="W. H. Bragg and W. L. Bragg, The reflection of X-rays by crystals, Proc. R. Soc. A 88, 428-438 (1913), doi:10.1098/rspa.1913.0040."),
    dict(family_id="bohr_atom", law_name="Bohr hydrogen energy levels",
         first_valid_year=1913, domain="quantum_and_atomic", generator="bohr_energy",
         equation="abs(e(n)) = a/n^2", x_range=[1.0, 24.0], sampling="linear",
         shape_class="declared_dictionary_overlap", signature="inverse_square",
         citation="N. Bohr, On the constitution of atoms and molecules, Philos. Mag. 26, 1-25 (1913), doi:10.1080/14786441308634955."),
    dict(family_id="moseley_xray", law_name="Moseley X-ray frequency law",
         first_valid_year=1913, domain="quantum_and_atomic", generator="moseley_law",
         equation="nu(z) = a (z-b)^2", x_range=[5.0, 82.0], sampling="linear",
         shape_class="smooth_composite", signature="shifted_quadratic",
         citation="H. G. J. Moseley, The high-frequency spectra of the elements, Philos. Mag. 26, 1024-1034 (1913), doi:10.1080/14786441308635052."),
    dict(family_id="langmuir_adsorption", law_name="Langmuir adsorption isotherm",
         first_valid_year=1918, domain="surface_physics", generator="langmuir_adsorption",
         equation="theta(x) = x/(1+x)", x_range=[0.03, 15.0], sampling="log",
         shape_class="smooth_composite", signature="rational_saturation",
         citation="I. Langmuir, The adsorption of gases on plane surfaces of glass, mica and platinum, J. Am. Chem. Soc. 40, 1361-1403 (1918), doi:10.1021/ja02242a004."),
    dict(family_id="saha_ionization", law_name="Saha ionization relation",
         first_valid_year=1920, domain="astrophysics", generator="saha_ionization",
         equation="r(t) = t^(3/2) exp(-a/t)", x_range=[0.12, 3.5], sampling="linear",
         shape_class="smooth_composite", signature="power_times_inverse_temperature_exponential",
         citation="M. N. Saha, Ionisation in the solar chromosphere, Nature 105, 232-233 (1920), doi:10.1038/105232b0."),
    dict(family_id="friedmann_expansion", law_name="Friedmann expansion equation",
         first_valid_year=1922, domain="cosmology", generator="friedmann_expansion",
         equation="h(z) = sqrt(omega_m (1+z)^3 + omega_k (1+z)^2 + omega_l)", x_range=[0.0, 6.0], sampling="linear",
         shape_class="smooth_composite", signature="square_root_polynomial_sum",
         citation="A. Friedmann, Uber die Krummung des Raumes, Z. Phys. 10, 377-386 (1922), doi:10.1007/BF01332580."),
    dict(family_id="compton_scattering", law_name="Compton wavelength shift",
         first_valid_year=1923, domain="quantum_and_atomic", generator="compton_shift",
         equation="delta(theta) = a (1-cos(theta))", x_range=[0.0, 3.141592653589793], sampling="linear",
         shape_class="smooth_composite", signature="one_minus_cosine_half_cycle",
         citation="A. H. Compton, A quantum theory of the scattering of X-rays by light elements, Phys. Rev. 21, 483-502 (1923), doi:10.1103/PhysRev.21.483."),
    dict(family_id="debye_huckel_screening", law_name="Debye-Huckel screened potential",
         first_valid_year=1923, domain="statistical_physics", generator="debye_huckel",
         equation="phi(r) = exp(-k r)/r", x_range=[0.08, 8.0], sampling="log",
         shape_class="smooth_composite", signature="screened_inverse",
         citation="P. Debye and E. Huckel, Zur Theorie der Elektrolyte. I, Phys. Z. 24, 185-206 (1923)."),
    dict(family_id="de_broglie_matter_wave", law_name="de Broglie wavelength",
         first_valid_year=1924, domain="quantum_and_atomic", generator="de_broglie",
         equation="lambda(p) = a/p", x_range=[0.08, 8.0], sampling="log",
         shape_class="declared_dictionary_overlap", signature="inverse",
         citation="L. de Broglie, Recherches sur la theorie des quanta, doctoral thesis, University of Paris (1924)."),
    dict(family_id="bose_einstein_statistics", law_name="Bose-Einstein occupation",
         first_valid_year=1924, domain="quantum_statistics", generator="bose_einstein",
         equation="n(x) = 1/(exp(x)-1)", x_range=[0.10, 7.0], sampling="log",
         shape_class="smooth_composite", signature="bose_occupation",
         citation="S. N. Bose, Plancks Gesetz und Lichtquantenhypothese, Z. Phys. 26, 178-181 (1924), doi:10.1007/BF01327326."),
    dict(family_id="fermi_dirac_statistics", law_name="Fermi-Dirac occupation",
         first_valid_year=1926, domain="quantum_statistics", generator="fermi_dirac",
         equation="n(x) = 1/(exp(x)+1)", x_range=[-7.0, 7.0], sampling="linear",
         shape_class="smooth_composite", signature="fermi_sigmoid",
         citation="E. Fermi, Zur Quantelung des idealen einatomigen Gases, Z. Phys. 36, 902-912 (1926), doi:10.1007/BF01400221."),
    dict(family_id="brillouin_paramagnetism", law_name="Brillouin magnetization function",
         first_valid_year=1927, domain="quantum_statistics", generator="brillouin_function",
         equation="b_j(x) = ((2j+1)/(2j))coth((2j+1)x/(2j))-(1/(2j))coth(x/(2j))", x_range=[0.05, 10.0], sampling="linear",
         shape_class="smooth_composite", signature="brillouin_coth_difference",
         citation="L. Brillouin, Les moments de rotation et le magnetisme dans la mecanique ondulatoire, J. Phys. Radium 8, 74-84 (1927), doi:10.1051/jphysrad:0192700802007400."),
    dict(family_id="gamow_tunnelling", law_name="Gamow barrier-penetration factor",
         first_valid_year=1928, domain="nuclear_physics", generator="gamow_factor",
         equation="p(e) = exp(-a/sqrt(e))", x_range=[0.08, 6.0], sampling="linear",
         shape_class="smooth_composite", signature="inverse_root_exponential",
         citation="G. Gamow, Zur Quantentheorie des Atomkernes, Z. Phys. 51, 204-212 (1928), doi:10.1007/BF01343196."),
    dict(family_id="fowler_nordheim_emission", law_name="Fowler-Nordheim field-emission law",
         first_valid_year=1928, domain="quantum_transport", generator="fowler_nordheim",
         equation="j(f) = f^2 exp(-a/f)", x_range=[0.18, 5.0], sampling="linear",
         shape_class="smooth_composite", signature="power_times_inverse_field_exponential",
         citation="R. H. Fowler and L. Nordheim, Electron emission in intense electric fields, Proc. R. Soc. A 119, 173-181 (1928), doi:10.1098/rspa.1928.0091."),
    dict(family_id="hubble_relation", law_name="Hubble distance-velocity relation",
         first_valid_year=1929, domain="cosmology", generator="hubble_law",
         equation="v(d) = h d", x_range=[0.05, 6.0], sampling="linear",
         shape_class="declared_dictionary_overlap", signature="linear",
         citation="E. Hubble, A relation between distance and radial velocity among extra-galactic nebulae, Proc. Natl. Acad. Sci. USA 15, 168-173 (1929), doi:10.1073/pnas.15.3.168."),
    dict(family_id="bethe_stopping", law_name="Bethe charged-particle stopping relation",
         first_valid_year=1930, domain="nuclear_physics", generator="bethe_stopping",
         equation="s(beta) = log(1+a beta^2/(1-beta^2))/beta^2-beta^2", x_range=[0.08, 0.96], sampling="linear",
         shape_class="smooth_composite", signature="relativistic_log_over_square",
         citation="H. Bethe, Zur Theorie des Durchgangs schneller Korpuskularstrahlen durch Materie, Ann. Phys. 397, 325-400 (1930), doi:10.1002/andp.19303970303."),
    dict(family_id="bloch_spin_wave", law_name="Bloch T^(3/2) magnetization law",
         first_valid_year=1930, domain="condensed_matter", generator="bloch_law",
         equation="m(t) = 1-a t^(3/2)", x_range=[0.02, 1.0], sampling="linear",
         shape_class="declared_dictionary_overlap", signature="offset_power_3_over_2",
         citation="F. Bloch, Zur Theorie des Ferromagnetismus, Z. Phys. 61, 206-219 (1930), doi:10.1007/BF01339661."),
    dict(family_id="london_two_fluid_penetration", law_name="Two-fluid/London penetration-depth relation",
         first_valid_year=1935, domain="condensed_matter", generator="london_penetration",
         equation="lambda(t) = (1-t^4)^(-1/2)", x_range=[0.04, 0.985], sampling="linear",
         shape_class="smooth_composite", signature="critical_inverse_square_root",
         citation="F. London and H. London, The electromagnetic equations of the supraconductor, Proc. R. Soc. A 149, 71-88 (1935), doi:10.1098/rspa.1935.0048; sampled with the contemporary two-fluid temperature law."),
    dict(family_id="breit_wigner_resonance", law_name="Breit-Wigner resonance profile",
         first_valid_year=1936, domain="nuclear_physics", generator="breit_wigner",
         equation="sigma(e) = a/((e-e0)^2+gamma^2)", x_range=[-4.0, 4.0], sampling="linear",
         shape_class="nonmonotone_or_oscillatory", signature="lorentzian_resonance",
         citation="G. Breit and E. Wigner, Capture of slow neutrons, Phys. Rev. 49, 519-531 (1936), doi:10.1103/PhysRev.49.519."),
    dict(family_id="rabi_oscillation", law_name="Rabi transition probability",
         first_valid_year=1937, domain="quantum_dynamics", generator="rabi_oscillation",
         equation="p(t) = sin^2(omega t/2)", x_range=[0.0, 18.84955592153876], sampling="linear",
         shape_class="nonmonotone_or_oscillatory", signature="sinusoidal_probability",
         citation="I. I. Rabi, Space quantization in a gyrating magnetic field, Phys. Rev. 51, 652-654 (1937), doi:10.1103/PhysRev.51.652."),
    dict(family_id="avrami_kinetics", law_name="Avrami transformation kinetics",
         first_valid_year=1939, domain="materials_physics", generator="avrami_kinetics",
         equation="x(t) = 1-exp(-k t^n)", x_range=[0.0, 3.5], sampling="linear",
         shape_class="smooth_composite", signature="stretched_exponential_saturation",
         citation="M. Avrami, Kinetics of phase change. I, J. Chem. Phys. 7, 1103-1112 (1939), doi:10.1063/1.1750380."),
    dict(family_id="kramers_escape", law_name="Kramers activated escape rate",
         first_valid_year=1940, domain="statistical_physics", generator="kramers_escape",
         equation="r(t) = a exp(-b/t)", x_range=[0.12, 3.0], sampling="linear",
         shape_class="smooth_composite", signature="inverse_temperature_exponential",
         citation="H. A. Kramers, Brownian motion in a field of force and the diffusion model of chemical reactions, Physica 7, 284-304 (1940), doi:10.1016/S0031-8914(40)90098-2."),
    dict(family_id="kolmogorov_turbulence", law_name="Kolmogorov inertial-range spectrum",
         first_valid_year=1941, domain="fluid_dynamics", generator="kolmogorov_spectrum",
         equation="e(k) = a k^(-5/3)", x_range=[0.08, 20.0], sampling="log",
         shape_class="declared_dictionary_overlap", signature="power_minus_5_over_3",
         citation="A. N. Kolmogorov, The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers, Dokl. Akad. Nauk SSSR 30, 301-305 (1941)."),
    dict(family_id="ising_spontaneous_magnetization", law_name="Yang exact 2D-Ising spontaneous magnetization",
         first_valid_year=1952, domain="statistical_physics", generator="ising_magnetization",
         equation="m(t) = max(1-sinh(2 k_c/t)^(-4),0)^(1/8)", x_range=[0.18, 1.30], sampling="linear",
         shape_class="threshold_or_piecewise", signature="critical_special_function",
         citation="C. N. Yang, The spontaneous magnetization of a two-dimensional Ising model, Phys. Rev. 85, 808-816 (1952), doi:10.1103/PhysRev.85.808."),
    dict(family_id="shockley_diode", law_name="Shockley ideal-diode equation",
         first_valid_year=1949, domain="semiconductor_physics", generator="shockley_diode",
         equation="i(v) = exp(v)-1", x_range=[-2.0, 5.5], sampling="linear",
         shape_class="declared_dictionary_overlap", signature="offset_exponential",
         citation="W. Shockley, The theory of p-n junctions in semiconductors and p-n junction transistors, Bell Syst. Tech. J. 28, 435-489 (1949), doi:10.1002/j.1538-7305.1949.tb03645.x."),
    dict(family_id="ginzburg_landau_order", law_name="Ginzburg-Landau mean-field order parameter",
         first_valid_year=1950, domain="condensed_matter", generator="ginzburg_landau",
         equation="psi(t) = sqrt(max(1-t,0))", x_range=[0.05, 1.35], sampling="linear",
         shape_class="threshold_or_piecewise", signature="critical_square_root_threshold",
         citation="V. L. Ginzburg and L. D. Landau, On the theory of superconductivity, Zh. Eksp. Teor. Fiz. 20, 1064-1082 (1950)."),
    dict(family_id="salpeter_initial_mass", law_name="Salpeter stellar initial-mass function",
         first_valid_year=1955, domain="astrophysics", generator="salpeter_imf",
         equation="phi(m) = a m^(-2.35)", x_range=[0.10, 30.0], sampling="log",
         shape_class="declared_dictionary_overlap", signature="power_minus_2_35",
         citation="E. E. Salpeter, The luminosity function and stellar evolution, Astrophys. J. 121, 161-167 (1955), doi:10.1086/145971."),
    dict(family_id="williams_landel_ferry", law_name="Williams-Landel-Ferry shift factor",
         first_valid_year=1955, domain="materials_physics", generator="wlf_shift",
         equation="log10(a_t) = -c1 dt/(c2+dt)", x_range=[-20.0, 100.0], sampling="linear",
         shape_class="smooth_composite", signature="rational_log_shift",
         citation="M. L. Williams, R. F. Landel and J. D. Ferry, The temperature dependence of relaxation mechanisms in amorphous polymers and other glass-forming liquids, J. Am. Chem. Soc. 77, 3701-3707 (1955), doi:10.1021/ja01619a008."),
    dict(family_id="bcs_superconducting_gap", law_name="BCS superconducting gap curve",
         first_valid_year=1957, domain="condensed_matter", generator="bcs_gap",
         equation="delta(t) ~= tanh(1.74 sqrt(1/t-1)) below tc; zero above", x_range=[0.05, 1.25], sampling="linear",
         shape_class="threshold_or_piecewise", signature="bcs_gap_interpolation",
         citation="J. Bardeen, L. N. Cooper and J. R. Schrieffer, Theory of superconductivity, Phys. Rev. 108, 1175-1204 (1957), doi:10.1103/PhysRev.108.1175; the sampled closed form is the standard reduced-temperature interpolation."),
    dict(family_id="josephson_current_phase", law_name="Josephson current-phase relation",
         first_valid_year=1962, domain="condensed_matter", generator="josephson_relation",
         equation="i(phi) = i_c sin(phi)", x_range=[-6.283185307179586, 6.283185307179586], sampling="linear",
         shape_class="nonmonotone_or_oscillatory", signature="sine",
         citation="B. D. Josephson, Possible new effects in superconductive tunnelling, Phys. Lett. 1, 251-253 (1962), doi:10.1016/0031-9163(62)91369-0."),
    dict(family_id="neutrino_oscillation", law_name="Two-flavour neutrino survival probability",
         first_valid_year=1962, domain="particle_physics", generator="neutrino_oscillation",
         equation="p(e) = 1-a sin^2(b/e)", x_range=[0.18, 6.0], sampling="linear",
         shape_class="nonmonotone_or_oscillatory", signature="inverse_argument_oscillation",
         citation="Z. Maki, M. Nakagawa and S. Sakata, Remarks on the unified model of elementary particles, Prog. Theor. Phys. 28, 870-880 (1962), doi:10.1143/PTP.28.870."),
    dict(family_id="kondo_resistivity", law_name="Kondo logarithmic resistivity",
         first_valid_year=1964, domain="condensed_matter", generator="kondo_resistivity",
         equation="rho(t) = rho0+c log(t0/t)", x_range=[0.05, 3.0], sampling="log",
         shape_class="smooth_composite", signature="logarithmic",
         citation="J. Kondo, Resistance minimum in dilute magnetic alloys, Prog. Theor. Phys. 32, 37-49 (1964), doi:10.1143/PTP.32.37."),
    dict(family_id="mott_variable_range_hopping", law_name="Mott variable-range hopping",
         first_valid_year=1968, domain="condensed_matter", generator="mott_vrh",
         equation="sigma(t) = exp(-(t0/t)^(1/4))", x_range=[0.02, 3.0], sampling="log",
         shape_class="smooth_composite", signature="stretched_inverse_temperature_exponential",
         citation="N. F. Mott, Conduction in non-crystalline materials, Philos. Mag. 19, 835-852 (1969); see also J. Non-Cryst. Solids 1, 1-17 (1968), doi:10.1016/0022-3093(68)90002-1."),
    dict(family_id="nauenberg_white_dwarf", law_name="Nauenberg white-dwarf mass-radius relation",
         first_valid_year=1972, domain="astrophysics", generator="white_dwarf_mass_radius",
         equation="r(m) = sqrt(m^(-2/3)-m^(2/3))", x_range=[0.08, 0.985], sampling="linear",
         shape_class="smooth_composite", signature="relativistic_mass_radius_turnover",
         citation="M. Nauenberg, Analytic approximations to the mass-radius relation and energy of zero-temperature stars, Astrophys. J. 175, 417-430 (1972), doi:10.1086/151568."),
    dict(family_id="bkt_correlation_length", law_name="Berezinskii-Kosterlitz-Thouless correlation length",
         first_valid_year=1973, domain="condensed_matter", generator="bkt_correlation_length",
         equation="xi(t) = exp(b/sqrt(t-1))", x_range=[1.08, 3.5], sampling="linear",
         shape_class="smooth_composite", signature="essential_singularity",
         citation="J. M. Kosterlitz and D. J. Thouless, Ordering, metastability and phase transitions in two-dimensional systems, J. Phys. C 6, 1181-1203 (1973), doi:10.1088/0022-3719/6/7/010."),
    dict(family_id="qcd_asymptotic_freedom", law_name="One-loop QCD running coupling",
         first_valid_year=1973, domain="particle_physics", generator="qcd_running",
         equation="alpha(q) = a/log(q^2)", x_range=[1.12, 30.0], sampling="log",
         shape_class="smooth_composite", signature="inverse_logarithm",
         citation="D. J. Gross and F. Wilczek, Ultraviolet behavior of non-Abelian gauge theories, Phys. Rev. Lett. 30, 1343-1346 (1973), doi:10.1103/PhysRevLett.30.1343."),
    dict(family_id="hawking_temperature", law_name="Hawking black-hole temperature",
         first_valid_year=1974, domain="gravitation", generator="hawking_temperature",
         equation="t(m) = a/m", x_range=[0.08, 20.0], sampling="log",
         shape_class="declared_dictionary_overlap", signature="inverse",
         citation="S. W. Hawking, Black hole explosions?, Nature 248, 30-31 (1974), doi:10.1038/248030a0."),
    dict(family_id="press_schechter_mass_function", law_name="Press-Schechter multiplicity function",
         first_valid_year=1974, domain="cosmology", generator="press_schechter",
         equation="f(nu) = a nu exp(-nu^2/2)", x_range=[0.03, 5.0], sampling="linear",
         shape_class="nonmonotone_or_oscillatory", signature="gaussian_weighted_power",
         citation="W. H. Press and P. Schechter, Formation of galaxies and clusters of galaxies by self-similar gravitational condensation, Astrophys. J. 187, 425-438 (1974), doi:10.1086/152650."),
    dict(family_id="schechter_luminosity", law_name="Schechter galaxy luminosity function",
         first_valid_year=1976, domain="astrophysics", generator="schechter_function",
         equation="phi(x) = a x^alpha exp(-x)", x_range=[0.03, 8.0], sampling="log",
         shape_class="nonmonotone_or_oscillatory", signature="power_times_exponential_cutoff",
         citation="P. Schechter, An analytic expression for the luminosity function for galaxies, Astrophys. J. 203, 297-306 (1976), doi:10.1086/154079."),
    dict(family_id="tully_fisher_relation", law_name="Tully-Fisher luminosity-velocity law",
         first_valid_year=1977, domain="astrophysics", generator="tully_fisher",
         equation="l(v) = a v^4", x_range=[0.08, 6.0], sampling="linear",
         shape_class="declared_dictionary_overlap", signature="power_4",
         citation="R. B. Tully and J. R. Fisher, A new method of determining distances to galaxies, Astron. Astrophys. 54, 661-673 (1977)."),
    dict(family_id="integer_quantum_hall", law_name="Integer quantum-Hall conductance staircase",
         first_valid_year=1980, domain="condensed_matter", generator="quantum_hall_steps",
         equation="g(x) = floor(x)", x_range=[0.05, 8.95], sampling="linear",
         shape_class="threshold_or_piecewise", signature="integer_staircase",
         citation="K. v. Klitzing, G. Dorda and M. Pepper, New method for high-accuracy determination of the fine-structure constant based on quantized Hall resistance, Phys. Rev. Lett. 45, 494-497 (1980), doi:10.1103/PhysRevLett.45.494."),
    dict(family_id="navarro_frenk_white", law_name="Navarro-Frenk-White density profile",
         first_valid_year=1996, domain="cosmology", generator="nfw_profile",
         equation="rho(x) = a/(x(1+x)^2)", x_range=[0.03, 30.0], sampling="log",
         shape_class="smooth_composite", signature="double_power_rational",
         citation="J. F. Navarro, C. S. Frenk and S. D. M. White, The structure of cold dark matter halos, Astrophys. J. 462, 563-575 (1996), doi:10.1086/177173."),
]


def coth(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.tanh(x)


def debye_integral(limit: float) -> float:
    def integrand(u: float) -> float:
        if u < 1e-5:
            return u * u
        if u > 80:
            return u**4 * np.exp(-u)
        return u**4 * np.exp(-u) / (1.0 - np.exp(-u)) ** 2
    return quad(integrand, 0.0, float(limit), epsabs=2e-8, epsrel=2e-8, limit=150)[0]


def evaluate(generator: str, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    a = float(rng.uniform(0.80, 1.20))
    if generator == "richardson_dushman":
        barrier = float(rng.uniform(1.6, 2.4)); y = a * x**2 * np.exp(-barrier / x); p = {"a": a, "barrier": barrier}
    elif generator == "radioactive_decay":
        rate = float(rng.uniform(0.65, 1.35)); y = a * np.exp(-rate * x); p = {"a": a, "rate": rate}
    elif generator == "lorentz_gamma":
        y = a / np.sqrt(1 - x**2); p = {"a": a}
    elif generator == "photoelectric_threshold":
        threshold = float(rng.uniform(0.85, 1.15)); y = a * np.maximum(x - threshold, 0); p = {"a": a, "threshold": threshold}
    elif generator == "brownian_msd":
        y = a * x; p = {"a": a}
    elif generator == "langevin_function":
        scale = float(rng.uniform(0.8, 1.2)); z = scale * x; y = a * (coth(z) - 1 / z); p = {"a": a, "scale": scale}
    elif generator == "einstein_heat_capacity":
        scale = float(rng.uniform(0.85, 1.15)); z = scale * x; y = a * z**2 * np.exp(-z) / (1 - np.exp(-z)) ** 2; p = {"a": a, "scale": scale}
    elif generator == "curie_weiss":
        critical = float(rng.uniform(0.94, 1.0)); y = a / (x - critical); p = {"a": a, "critical": critical}
    elif generator == "hill_binding":
        hill = float(rng.uniform(2.2, 4.2)); half = float(rng.uniform(0.8, 1.2)); y = a * x**hill / (half**hill + x**hill); p = {"a": a, "hill": hill, "half": half}
    elif generator == "geiger_nuttall":
        barrier = float(rng.uniform(2.5, 4.5)); y = a * np.exp(-barrier / np.sqrt(x)); p = {"a": a, "barrier": barrier}
    elif generator == "rutherford_scattering":
        y = a / np.sin(x / 2) ** 4; p = {"a": a}
    elif generator == "child_langmuir":
        y = a * x**1.5; p = {"a": a}
    elif generator == "debye_heat_capacity":
        scale = float(rng.uniform(0.88, 1.12)); t = x / scale; y = a * np.asarray([9 * value**3 * debye_integral(1 / value) for value in t]); p = {"a": a, "theta_scale": scale}
    elif generator == "bragg_diffraction":
        scale = float(rng.uniform(0.90, 1.0)); y = a * np.arcsin(np.clip(scale * x, -0.999999, 0.999999)); p = {"a": a, "scale": scale}
    elif generator == "bohr_energy":
        y = a / x**2; p = {"a": a}
    elif generator == "moseley_law":
        screening = float(rng.uniform(0.7, 1.3)); y = a * (x - screening) ** 2; p = {"a": a, "screening": screening}
    elif generator == "langmuir_adsorption":
        affinity = float(rng.uniform(0.7, 1.4)); y = a * affinity * x / (1 + affinity * x); p = {"a": a, "affinity": affinity}
    elif generator == "saha_ionization":
        barrier = float(rng.uniform(1.6, 2.5)); y = a * x**1.5 * np.exp(-barrier / x); p = {"a": a, "barrier": barrier}
    elif generator == "friedmann_expansion":
        omega_m = float(rng.uniform(0.24, 0.38)); omega_k = float(rng.uniform(-0.04, 0.04)); omega_l = 1 - omega_m - omega_k; y = a * np.sqrt(omega_m * (1 + x)**3 + omega_k * (1 + x)**2 + omega_l); p = {"a": a, "omega_m": omega_m, "omega_k": omega_k, "omega_l": omega_l}
    elif generator == "compton_shift":
        y = a * (1 - np.cos(x)); p = {"a": a}
    elif generator == "debye_huckel":
        screening = float(rng.uniform(0.7, 1.3)); y = a * np.exp(-screening * x) / x; p = {"a": a, "screening": screening}
    elif generator == "de_broglie":
        y = a / x; p = {"a": a}
    elif generator == "bose_einstein":
        scale = float(rng.uniform(0.85, 1.15)); y = a / np.expm1(scale * x); p = {"a": a, "scale": scale}
    elif generator == "fermi_dirac":
        shift = float(rng.uniform(-0.35, 0.35)); y = a / (np.exp(np.clip(x - shift, -60, 60)) + 1); p = {"a": a, "chemical_shift": shift}
    elif generator == "brillouin_function":
        j = float(rng.choice([0.5, 1.0, 1.5, 2.0, 2.5])); z1 = (2*j+1)*x/(2*j); z2 = x/(2*j); y = a * ((2*j+1)/(2*j)*coth(z1) - 1/(2*j)*coth(z2)); p = {"a": a, "j": j}
    elif generator == "gamow_factor":
        barrier = float(rng.uniform(2.2, 4.2)); y = a * np.exp(-barrier / np.sqrt(x)); p = {"a": a, "barrier": barrier}
    elif generator == "fowler_nordheim":
        barrier = float(rng.uniform(1.8, 3.2)); y = a * x**2 * np.exp(-barrier / x); p = {"a": a, "barrier": barrier}
    elif generator == "hubble_law":
        y = a * x; p = {"a": a}
    elif generator == "bethe_stopping":
        strength = float(rng.uniform(12.0, 24.0)); y = a * (np.log1p(strength*x**2/(1-x**2))/x**2 - x**2); p = {"a": a, "strength": strength}
    elif generator == "bloch_law":
        reduction = float(rng.uniform(0.25, 0.55)); y = a * (1 - reduction * x**1.5); p = {"a": a, "reduction": reduction}
    elif generator == "london_penetration":
        y = a / np.sqrt(1 - x**4); p = {"a": a}
    elif generator == "breit_wigner":
        center = float(rng.uniform(-0.35, 0.35)); width = float(rng.uniform(0.25, 0.55)); y = a / ((x-center)**2 + width**2); p = {"a": a, "center": center, "width": width}
    elif generator == "rabi_oscillation":
        frequency = float(rng.uniform(0.85, 1.15)); y = a * np.sin(frequency*x/2)**2; p = {"a": a, "frequency": frequency}
    elif generator == "avrami_kinetics":
        exponent = float(rng.uniform(1.8, 3.8)); rate = float(rng.uniform(0.65, 1.15)); y = a * (1 - np.exp(-rate*x**exponent)); p = {"a": a, "exponent": exponent, "rate": rate}
    elif generator == "kramers_escape":
        barrier = float(rng.uniform(1.8, 2.8)); y = a * np.exp(-barrier/x); p = {"a": a, "barrier": barrier}
    elif generator == "kolmogorov_spectrum":
        y = a * x**(-5/3); p = {"a": a}
    elif generator == "ising_magnetization":
        kc = 0.5 * np.log(1 + np.sqrt(2)); k = kc / x; inside = 1 - np.sinh(2*k)**-4; y = a * np.where(x < 1, np.maximum(inside, 0)**0.125, 0); p = {"a": a, "kc": float(kc)}
    elif generator == "shockley_diode":
        ideality = float(rng.uniform(0.85, 1.20)); y = a * (np.exp(np.clip(x/ideality, -60, 60)) - 1); p = {"a": a, "ideality": ideality}
    elif generator == "ginzburg_landau":
        critical = float(rng.uniform(0.95, 1.05)); y = a * np.sqrt(np.maximum(1 - x/critical, 0)); p = {"a": a, "critical": critical}
    elif generator == "salpeter_imf":
        exponent = float(rng.uniform(2.25, 2.45)); y = a * x**(-exponent); p = {"a": a, "exponent": exponent}
    elif generator == "wlf_shift":
        c1 = float(rng.uniform(14.0, 19.0)); c2 = float(rng.uniform(45.0, 60.0)); y = -c1*x/(c2+x); p = {"c1": c1, "c2": c2}
    elif generator == "bcs_gap":
        critical = float(rng.uniform(0.96, 1.04)); reduced = x/critical; y = np.where(reduced < 1, a*np.tanh(1.74*np.sqrt(np.maximum(1/reduced-1, 0))), 0); p = {"a": a, "critical": critical}
    elif generator == "josephson_relation":
        phase = float(rng.uniform(-0.25, 0.25)); y = a * np.sin(x + phase); p = {"a": a, "phase": phase}
    elif generator == "neutrino_oscillation":
        amplitude = float(rng.uniform(0.65, 0.98)); frequency = float(rng.uniform(1.5, 3.0)); y = 1 - amplitude*np.sin(frequency/x)**2; p = {"amplitude": amplitude, "frequency": frequency}
    elif generator == "kondo_resistivity":
        strength = float(rng.uniform(0.12, 0.28)); y = a + strength*np.log(1/x); p = {"offset": a, "strength": strength}
    elif generator == "mott_vrh":
        scale = float(rng.uniform(0.75, 1.35)); y = a * np.exp(-(scale/x)**0.25); p = {"a": a, "scale": scale}
    elif generator == "white_dwarf_mass_radius":
        y = a * np.sqrt(np.maximum(x**(-2/3)-x**(2/3), 0)); p = {"a": a}
    elif generator == "bkt_correlation_length":
        strength = float(rng.uniform(0.8, 1.3)); y = a * np.exp(strength/np.sqrt(x-1)); p = {"a": a, "strength": strength}
    elif generator == "qcd_running":
        beta = float(rng.uniform(0.75, 1.25)); y = a / (beta*np.log(x**2)); p = {"a": a, "beta": beta}
    elif generator == "hawking_temperature":
        y = a / x; p = {"a": a}
    elif generator == "press_schechter":
        scale = float(rng.uniform(0.85, 1.15)); z = x/scale; y = a*z*np.exp(-z**2/2); p = {"a": a, "scale": scale}
    elif generator == "schechter_function":
        alpha = float(rng.uniform(-0.9, -0.45)); scale = float(rng.uniform(0.85, 1.15)); z = x/scale; y = a*z**alpha*np.exp(-z); p = {"a": a, "alpha": alpha, "scale": scale}
    elif generator == "tully_fisher":
        exponent = float(rng.uniform(3.6, 4.4)); y = a*x**exponent; p = {"a": a, "exponent": exponent}
    elif generator == "quantum_hall_steps":
        shift = float(rng.uniform(-0.2, 0.2)); y = a*np.floor(np.maximum(x+shift, 0)); p = {"a": a, "shift": shift}
    elif generator == "nfw_profile":
        scale = float(rng.uniform(0.8, 1.2)); z = x/scale; y = a/(z*(1+z)**2); p = {"a": a, "scale": scale}
    else:
        raise KeyError(generator)
    return np.asarray(y, float), p


def sample_x(row: dict, rng: np.random.Generator) -> np.ndarray:
    lo, hi = map(float, row["x_range"])
    if row["sampling"] == "log":
        x = np.geomspace(lo, hi, POINTS)
    else:
        x = np.linspace(lo, hi, POINTS)
    # Jitter only interior locations and retain the declared support exactly.
    step = np.diff(x)
    local = np.r_[step[0], (step[:-1] + step[1:]) / 2, step[-1]]
    x = x + rng.normal(0, 0.10, POINTS) * local
    return np.sort(np.clip(x, lo, hi))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert len(REGISTRY) == len({row["family_id"] for row in REGISTRY})
    assert all(int(row["first_valid_year"]) > 1900 for row in REGISTRY)
    assert all(row["citation"].strip() for row in REGISTRY)
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    registry_path = DATA / "post1900_formula_registry.json"
    registry_payload = {
        "status": "citation-anchored curated post-1900 formula-level benchmark registry",
        "cutoff_policy": "first_valid_year > 1900; Planck 1900 is excluded",
        "provenance_policy": "Every family has a named primary historical citation; formula-to-source fidelity is a manual bibliographic audit, not a claim of machine-verified transcription.",
        "inference_unit": "source-law family",
        "representation": "one-dimensional nondimensional reduction named in equation",
        "reduction_notes": {
            "special_relativity": "The relativistic physical interpretation is dated to 1905; Lorentz-factor mathematics has pre-1905 antecedents, making this an intentionally hard horizon case.",
            "friedmann_expansion": "A standard matter-curvature-Lambda one-dimensional specialization of the Friedmann equation is sampled.",
            "london_two_fluid_penetration": "The temperature dependence combines the 1934 two-fluid ansatz with London electrodynamics; it is not a literal transcription of one equation in the cited London paper.",
            "bcs_superconducting_gap": "The standard closed-form interpolation to the BCS reduced gap is sampled rather than numerically solving the implicit gap equation.",
            "neutrino_oscillation": "The standard two-flavour survival-probability reduction of the mixing framework is sampled.",
            "integer_quantum_hall": "An ideal conductance staircase versus filling factor is sampled; plateau broadening and localization physics are omitted.",
        },
        "claim_boundary": "Formula-level temporal transfer; not real-measurement discovery and not publication-year classification.",
        "laws": REGISTRY,
    }
    registry_path.write_text(json.dumps(registry_payload, indent=2) + "\n", encoding="utf-8")

    clouds, family_ids, record_ids, metadata = [], [], [], []
    for family_index, row in enumerate(REGISTRY):
        for repeat in range(REPEATS):
            record_seed = SEED + 1000*family_index + repeat
            rng = np.random.default_rng(record_seed)
            x = sample_x(row, rng)
            y_clean, parameters = evaluate(row["generator"], x, rng)
            assert np.all(np.isfinite(y_clean)), (row["family_id"], repeat)
            noise_fraction = float(rng.uniform(0.005, 0.025))
            scale = max(float(np.std(y_clean)), 1e-10)
            hetero = 0.75 + 0.50*(x-x.min())/(max(float(np.ptp(x)), 1e-12))
            y = y_clean + rng.normal(0, noise_fraction*scale, POINTS)*hetero
            cloud = np.stack([x, y], axis=1).astype(np.float32)
            record_id = f"{row['family_id']}__r{repeat:02d}"
            clouds.append(cloud); family_ids.append(row["family_id"]); record_ids.append(record_id)
            metadata.append({
                "record_id": record_id, "family_id": row["family_id"], "repeat": repeat,
                "seed": record_seed, "parameters": parameters, "noise_fraction": noise_fraction,
                "n": POINTS, "sampling": row["sampling"], "x_range": row["x_range"],
            })

    archive_path = DATA / "post1900_formula_pointclouds.npz"
    np.savez_compressed(
        archive_path, X=np.asarray(clouds, np.float32),
        family_ids=np.asarray(family_ids), record_ids=np.asarray(record_ids),
    )
    records_path = DATA / "post1900_formula_records.jsonl"
    records_path.write_text("".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8")

    shape_counts = {}
    signature_counts = {}
    for row in REGISTRY:
        shape_counts[row["shape_class"]] = shape_counts.get(row["shape_class"], 0) + 1
        signature_counts[row["signature"]] = signature_counts.get(row["signature"], 0) + 1
    duplicate_signatures = {name: count for name, count in signature_counts.items() if count > 1}
    manifest = {
        "status": "PASS",
        "seed": SEED, "families": len(REGISTRY), "records": len(clouds),
        "records_per_family": REPEATS, "points_per_record": POINTS,
        "year_min": min(row["first_valid_year"] for row in REGISTRY),
        "year_max": max(row["first_valid_year"] for row in REGISTRY),
        "shape_class_counts": shape_counts,
        "duplicate_functional_signatures_retained_as_hard_temporal_cases": duplicate_signatures,
        "files": {
            registry_path.name: {"sha256": sha256(registry_path), "bytes": registry_path.stat().st_size},
            archive_path.name: {"sha256": sha256(archive_path), "bytes": archive_path.stat().st_size},
            records_path.name: {"sha256": sha256(records_path), "bytes": records_path.stat().st_size},
        },
        "leakage_control": "No benchmark record is used to train or calibrate either frozen EPOCH component.",
        "claim_boundary": registry_payload["claim_boundary"],
    }
    output = RESULTS / "post1900_formula_benchmark_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
