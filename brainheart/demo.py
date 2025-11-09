from brainheart.load_reference_dataset import load
from brainheart.loading.ecg_loading import annotate_valid_ecg_periods, load_ecg, identify_ecg_channel
from brainheart.wrappers.ecg_wrappers import (
    find_ecg_events_neurokit, 
    ecg_quality_sliding_window_zhao2018_neurokit, 
    ecg_clean_neurokit, 
    hr_neurokit2,
    ecg_delineate_neurokit2, 
    annotate_bradycardia, 
    ecg_fixpeaks_neurokit, 
    hr_corrected_neurokit
)

raw = load(0)

annotate_valid_ecg_periods(raw)
identify_ecg_channel(raw)
ecg = load_ecg(raw)
find_ecg_events_neurokit(raw)
ecg_quality_sliding_window_zhao2018_neurokit(
    raw,
    window_overlap_sec = 15, 
    keep_barely_acceptable=False)
ecg_clean_neurokit(raw)
rate = hr_neurokit2(raw, min_N_peaks = 1)
ecg_delineate_neurokit2(raw)
annotate_bradycardia(raw)
ecg_fixpeaks_neurokit(raw)
hr_corrected_neurokit(raw)