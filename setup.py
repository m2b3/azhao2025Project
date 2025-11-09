from setuptools import setup, find_packages

setup(
    name = "brainheart", 
    version = "0.1.0", 
    packages = find_packages(), 
    install_requires = [        
        "mne", 
        "mne_bids", 
        "numpy", 
        "scipy", 
        "pandas", 
        "neurokit2", 
        "pytest"
    ]
)