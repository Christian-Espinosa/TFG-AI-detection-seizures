import pandas as pd
import os
import numpy as np


import ctypes
import win32api
import win32security

def suspend(hibernate=False):
    """Puts Windows to Suspend/Sleep/Standby or Hibernate.

    Parameters
    ----------
    hibernate: bool, default False
        If False (default), system will enter Suspend/Sleep/Standby state.
        If True, system will Hibernate, but only if Hibernate is enabled in the
        system settings. If it's not, system will Sleep.

    Example:
    --------
    >>> suspend()
    """
    # Enable the SeShutdown privilege (which must be present in your
    # token in the first place)
    priv_flags = (win32security.TOKEN_ADJUST_PRIVILEGES |
                  win32security.TOKEN_QUERY)
    hToken = win32security.OpenProcessToken(
        win32api.GetCurrentProcess(),
        priv_flags
    )
    priv_id = win32security.LookupPrivilegeValue(
       None,
       win32security.SE_SHUTDOWN_NAME
    )
    old_privs = win32security.AdjustTokenPrivileges(
        hToken,
        0,
        [(priv_id, win32security.SE_PRIVILEGE_ENABLED)]
    )

    if (win32api.GetPwrCapabilities()['HiberFilePresent'] == False and
        hibernate == True):
            import warnings
            warnings.warn("Hibernate isn't available. Suspending.")
    try:
        ctypes.windll.powrprof.SetSuspendState(not hibernate, True, False)
    except:
        # True=> Standby; False=> Hibernate
        # https://msdn.microsoft.com/pt-br/library/windows/desktop/aa373206(v=vs.85).aspx
        # says the second parameter has no effect.
#        ctypes.windll.kernel32.SetSystemPowerState(not hibernate, True)
        win32api.SetSystemPowerState(not hibernate, True)

    # Restore previous privileges
    win32security.AdjustTokenPrivileges(
        hToken,
        0,
        old_privs
    )


#subj = 'chb' + "{:02.0f}".format(1)
#numpys = os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + '/numpy/')
            
#Numpys
#npys = os.listdir(numpys)
#for file in range(0,int(len(npys)),2):
#
#    print("Loading:".format(npys[file]))
#    data_x = np.load(os.path.join(numpys,npys[file]), allow_pickle=True)
#    data_y = np.load(os.path.join(numpys,npys[file+1]), allow_pickle=True)
#    print('file {}: {}'.format(npys[file], data_x.shape))
#    print('file {}: {}'.format(npys[file+1], data_y.shape))
#
#suspend(True)
subj = 'chb' + "{:02.0f}".format(12)
eeg_df = pd.read_parquet(os.path.abspath(os.path.join(os.getcwd(), os.pardir) + "/DataSetTFG/CHB-MIT/" + subj + '/results/' + 'chb12_08_data_x.parquet'))
print(eeg_df)