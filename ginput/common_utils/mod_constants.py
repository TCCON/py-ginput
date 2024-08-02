import os

avogadro = 6.022141e23  # molec./mol
mass_dry_air = 28.964e-3  # kg/mol
mass_h2o = 18.01534e-3  # kg/mol
gas_const = 8.314e4 # gas constant in cm^3 * hPa / (mol * K)

ratio_molec_mass = 28.964/18.02	 # Ratio of Molecular Masses (Dry_Air/H2O)

trop_flag = 1
middleworld_flag = 2
overworld_flag = 3

_mydir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(_mydir, '..', 'data'))

days_per_year = 365.25

priors_version = '1.0.0'

# US Standard Atmosphere
p_ussa = (10.0,  5.0,   2.0,   1.0,   0.5,    0.2,   0.1,   0.01,  0.001, 0.0001)
t_ussa = (227.7, 239.2, 257.9, 270.6, 264.3, 245.2, 231.6, 198.0, 189.8, 235.0)
z_ussa = (31.1,  36.8,  42.4,  47.8,  53.3,  60.1,  64.9,  79.3,  92.0,  106.3)
