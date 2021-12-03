"""
This module defines the global variables to be used along the program.

Updated by:
    elias 19-04-21

    How to use it?

    import eeg_globals as gbl

"""

dataset_collection = ['aslogic', 'cvc', 'uab']

aslogic_subjects = ['agusti', 'alejandro', 'cristian', 'dani', 'jose', 'qiang']
cvc_subjects = ['aura', 'carles', 'eliot', 'esmitt', 'guille', 'hector', 'lluis', 'sanket', 'thomas']
uab_subjects = ['alba', 'anastasia', 'gonzalo', 'joan', 'jordi', 'juanjo', 'judit', 'marsel', 'roger']

selected_subjects = ['agusti', 'alejandro', 'carles', 'cristian', 'dani', 'eliot',
                     'esmitt', 'gonzalo', 'guille', 'hector', 'jordi', 'jose',
                     'juanjo', 'lluis', 'sanket', 'thomas']

ds_subjects = aslogic_subjects + cvc_subjects + uab_subjects # dataset subjects

# =============================================================================
# definition for the EEG headset of 14 nodes
# =============================================================================

waves = ['theta', 'alpha', 'betal', 'betah', 'gamma']
dic_wave = {i : waves[i] for i in range(len(waves))}
dic_wave_id = {wave : idx for idx, wave in enumerate(waves)}

# Generic NODE_NAMES for a 14 headset according the 10/20 metrics
nodes = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
dic_node = {i : nodes[i] for i in range(len(nodes))} # A dict for IdPosition on the node
eeg_nodes = ['EEG.' + i for i in nodes]

quality_nodes = ['CQ.' + i for i in nodes] # DataFrame colums with the format CQ.NODE_NAME
user_metalabels = ['datetime', 'subject', 'test', 'phase'] # DataFrame columns that contains the pre-processed data
user_basic_metalabels = ['subject', 'test', 'phase']

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

all_pow_waves = list(dic_pow_waves.keys())


dic_rhythm_bandwidth = {    'delta' : [0.5, 4],
                            'theta' : [4, 8],
                            'alpha' : [8, 12],
                            'beta' : [12, 30],
                            'gamma' : [30, 45],
                        }


# adapted from
# https://www.emotiv.com/glossary/electroencephalogram/
# and from https://www.emotiv.com/knowledge-base/frequency-bands-what-are-they-and-how-do-i-access-them/
dic_eeg_rhythm_bandwidth = { 'theta' : [4, 8],
                             'alpha' : [8, 12],
                             'betal' : [12, 16],
                             'betah' : [16, 25],
                             'gamma' : [25, 45],
                              }




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

dic_colors = {1 : r'red',
             2 : r'green',
             3 : r'blue'
             }

# =============================================================================
# Sampling frequency
# =============================================================================
PSF = 8 # power spectral sampling frequency
RSF = 128 # rawdata sampling frequency

# =============================================================================
# workload code
# =============================================================================

CODE_BL = {11, 21, 31} # baseline
CODE_WL = {12, 22, 32} # workload
CODE_REC = {13, 23, 33} # recovery

dic_baseline = {11 : 'BL-1', 21 : 'BL-2', 31 : 'BL-3'}
dic_workload = {12 : 'WL-1', 22 : 'WL-2', 32 : 'WL-3'}


# =============================================================================
# windows and overlapping
# =============================================================================
dic_window = { 5 : 0,
              10 : 5,
              20 : 10,
              40 : 30
              }

"""

SOURCE:  https://www.emotiv.com/glossary/electroencephalogram/

What do Electroencephalograms Show?

Electroencephalograms collect electrical activity displayed by electroencephalogram waves. Applying  the Fast Fourier Transform (FFT) to the
raw brainwaves identifies four main frequencies: alpha, delta, beta and theta waves. Ranges in these different frequencies can be associated
with different states of mind and cognitive functions.

 * Beta Waves (frequency range from 14 Hz to about 30 Hz)

Beta waves are most closely associated with attention and alertness. When beta waves display a low-amplitude pattern, this is often associated
with active concentration.

* Alpha Waves (frequency range from 7 Hz to 13 Hz)

Alpha waves are often associated with a relaxed state of mind. They are not detected during more intense cognitive processes like thinking or
problem-solving.

 * Theta Waves (frequency range from 4 Hz to 7 Hz)

Theta waves are associated with memory formation and navigation. They typically occur during deep REM (rapid eye movement) sleep.
REM sleep begins in response to signals being sent to the brain’s cerebral cortex, which is responsible for learning, thinking and
organizing information.

* Delta Waves (frequency range up to 4 Hz)

Delta waves are large, slow brainwaves associated with stages 3 and 4 of non-REM sleep, the deepest and most restorative stages of sleep.
Stage 3 is also called “Delta sleep”.

"""


"""
SOURCE : https://www.emotiv.com/knowledge-base/frequency-bands-what-are-they-and-how-do-i-access-them/

Using the Emotiv Community SDK and Advanced SDK versions, users can access the power in different brain activity bands for each individual sensor, updated twice per second and calculated over the previous 2 seconds of activity. This data can be accessed using the API call:

IEE_GetAverageBandPowers(unsigned int userId, IEE_DataChannel_t channel,
                                 double* theta, double* alpha, double* low_beta, double* high_beta, double* gamma);

"""