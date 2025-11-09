import numpy as np
import pandas as pd

import mne

from brainheart.ecg_channel_names_enum import ECG_Channels

phys_metadata_columns = [
    ECG_Channels.ECG_Phase_Completion_Atrial.value, 
    ECG_Channels.ECG_Phase_Completion_Ventricular.value, 
    ECG_Channels.ECG_Rate.value, 
]

def event_phys_metadata(
        raw: mne.io.BaseRaw, 
        events: np.ndarray, 
        event_id: int | None = None, 
        phys_metadata_columns = phys_metadata_columns
): 
    if event_id is not None: 
        events = events[
            events[:, 2] == event_id, :
        ]
    event_indices = events[:, 0]
    # Now get all the appropriate metadata
    raw_data = raw.get_data(picks = phys_metadata_columns, return_times = False)
    event_metadata_columns = raw_data[:, event_indices]
    return pd.DataFrame(
        data = event_metadata_columns.T, 
        columns = phys_metadata_columns
    )
