"""SimSiam self-supervised pretraining for BraTS 2021 brain MRI.

Pipeline (per BrainTumor.docx):
    BraTS (T1, T1ce, T2, FLAIR)  ->  SimSiam SSL pretraining  ->  frozen encoder
    encoder  ->  256-d deep feature vector per subject  ->  downstream IDH classifier

This package only covers the SSL pretraining + feature-extraction stages.
"""

__version__ = "0.1.0"

# Silence one known-harmless third-party import-time deprecation so cluster logs stay clean.
# NARROW, message-matched filter only (real warnings are untouched); set here because the package
# is imported before the submodules import torch/monai — which is when this FutureWarning fires.
import warnings as _warnings

_warnings.filterwarnings("ignore", message=r".*cuda\.cudart.*", category=FutureWarning)
del _warnings
