"""
This module defines the global variables to be used along the program.

Updated by:
    elias 19-04-2021

"""

aslogic_subjects = ['agusti', 'alejandro', 'cristian', 'dani', 'jose', 'qiang']
cvc_subjects = ['aura', 'carles', 'eliot', 'esmitt', 'guille', 'hector', 'lluis', 'sanket', 'thomas']
all_subjects = aslogic_subjects + cvc_subjects

dic_aslogic_subject = {'agusti' : 'Subject 1',
               'alejandro'  : 'Subject 2',
               'cristian' : 'Subject 3',
               'dani' : 'Subject 4',
               'jose' : 'Subject 5',
               'qiang' : 'Subject 6',
               }

# definition for an EEG headset of 14 nodes

waves = ['theta', 'alpha', 'betal', 'betah', 'gamma']
dic_wave = {i : waves[i] for i in range(len(waves))}


nodes = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
dic_node = {i : nodes[i] for i in range(len(nodes))}

quality_nodes = ['CQ.' + i for i in nodes]
user_metalabels = ['datetime','subject', 'test', 'phase'] # OUR RAW DATA ALWAYS HAS THIS LABELS

eeg_nodes = ['EEG.' + i for i in nodes]

pow_theta_nodes = ['POW.AF3.Theta',
 'POW.F7.Theta',
 'POW.F3.Theta',
 'POW.FC5.Theta',
 'POW.T7.Theta',
 'POW.P7.Theta',
 'POW.O1.Theta',
 'POW.O2.Theta',
 'POW.P8.Theta',
 'POW.T8.Theta',
 'POW.FC6.Theta',
 'POW.F4.Theta',
 'POW.F8.Theta',
 'POW.AF4.Theta']

pow_alpha_nodes = ['POW.AF3.Alpha',
 'POW.F7.Alpha',
 'POW.F3.Alpha',
 'POW.FC5.Alpha',
 'POW.T7.Alpha',
 'POW.P7.Alpha',
 'POW.O1.Alpha',
 'POW.O2.Alpha',
 'POW.P8.Alpha',
 'POW.T8.Alpha',
 'POW.FC6.Alpha',
 'POW.F4.Alpha',
 'POW.F8.Alpha',
 'POW.AF4.Alpha']

pow_betal_nodes = ['POW.AF3.BetaL',
 'POW.F7.BetaL',
 'POW.F3.BetaL',
 'POW.FC5.BetaL',
 'POW.T7.BetaL',
 'POW.P7.BetaL',
 'POW.O1.BetaL',
 'POW.O2.BetaL',
 'POW.P8.BetaL',
 'POW.T8.BetaL',
 'POW.FC6.BetaL',
 'POW.F4.BetaL',
 'POW.F8.BetaL',
 'POW.AF4.BetaL']

pow_betah_nodes = ['POW.AF3.BetaH',
 'POW.F7.BetaH',
 'POW.F3.BetaH',
 'POW.FC5.BetaH',
 'POW.T7.BetaH',
 'POW.P7.BetaH',
 'POW.O1.BetaH',
 'POW.O2.BetaH',
 'POW.P8.BetaH',
 'POW.T8.BetaH',
 'POW.FC6.BetaH',
 'POW.F4.BetaH',
 'POW.F8.BetaH',
 'POW.AF4.BetaH']

pow_gamma_nodes = ['POW.AF3.Gamma',
 'POW.F7.Gamma',
 'POW.F3.Gamma',
 'POW.FC5.Gamma',
 'POW.T7.Gamma',
 'POW.P7.Gamma',
 'POW.O1.Gamma',
 'POW.O2.Gamma',
 'POW.P8.Gamma',
 'POW.T8.Gamma',
 'POW.FC6.Gamma',
 'POW.F4.Gamma',
 'POW.F8.Gamma',
 'POW.AF4.Gamma']

all_pow_nodes = pow_theta_nodes + pow_alpha_nodes +  pow_betal_nodes + pow_betah_nodes + pow_gamma_nodes

dic_pow_waves = {'theta' : pow_theta_nodes,
             'alpha' : pow_alpha_nodes,
             'betal' : pow_betal_nodes,
             'betah' : pow_betah_nodes,
             'gamma' : pow_gamma_nodes,
             }



dic_band_definitions = { 'delta' : [0.5, 4],
           'theta' : [4, 8],
           'alpha' : [8, 12],
           'beta' : [12, 30],
           'gamma' : [30, 45],
           }

all_pow_waves = list(dic_pow_waves.keys())
# =============================================================================
# colors for plots
# =============================================================================

subject_color = ['r', 'g', 'b', 'c', 'm', 'y']
subject_marker = ['1', '2', '3', '4', '+', 'x']

bl_linestyles = { 1 : '-', # solid
                 2 : '-.', # dash-dotted
                 3: ':', # dotted
                 -1: '--' # dashed
                 }

wl_colors = {1 : 'c',
             2 : 'm',
             3: 'y',
             -1: 'orange'
             }

bl_colors = {1 : 'r',
             2 : 'g',
             3: 'b',
             -1: 'purple'
             }

bl_labels = {1 : 'BL - 1',
             2 : 'BL - 2',
             3 : 'BL - 3'
             }

wl_labels = {1 : 'WL - 1',
             2 : 'WL - 2',
             3 : 'WL - 3'
             }


# =============================================================================
# Sampling frequency
# =============================================================================
PSF = 8.0 # power spectral sampling frequency
RSF = 128.0 # rawdata sampling frequency

# =============================================================================
# workload code
# =============================================================================

CODE_BL = {11, 21, 31} # baseline
CODE_WL = {12, 22, 32} # workload
CODE_REC = {13, 23, 33} # recovery

dic_baseline = {11 : 'BL-1', 21 : 'BL-2', 31 : 'BL-3'}
dic_workload = {12 : 'WL-1', 22 : 'WL-2', 32 : 'WL-3'}
