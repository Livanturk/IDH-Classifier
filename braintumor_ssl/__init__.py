"""SimSiam self-supervised pretraining for BraTS 2021 brain MRI.

Pipeline (per BrainTumor.docx):
    BraTS (T1, T1ce, T2, FLAIR)  ->  SimSiam SSL pretraining  ->  frozen encoder
    encoder  ->  256-d deep feature vector per subject  ->  downstream IDH classifier

This package only covers the SSL pretraining + feature-extraction stages.
"""

__version__ = "0.1.0"
