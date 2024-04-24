# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:01:38 2023

@author: Palkowima, henningfa, starkeseb
"""

# References
# Dutz, A. (2020).
# "Towards patient selection for cranial proton beam therapy Assessment of "
# "current patient individual treatment decision strategies"
# (Doctoral dissertation).
# Technische Universität Dresden, Dresden. p. 42, p.56, p.112

import scipy
import numpy as np
from let.data import fuse_rois

# -----------------------------------------------
# ----------- defining helper functions -------


def gEUD_calculator(d_i, v_i, a):
    # v_i = volume in a dose bin
    # d_i = dose in a bin i of differential dvh
    # a = volume parameter equal to 1/n
    # n = volume effect of considered OAR, seriell: n=0; paralell: n=1

    vd = v_i * d_i**a
    gEUD = np.sum(vd)**(1/a)
    return gEUD

# -----------------------------------------------

# Acute Tox

# Alopecia grade >=1 (CTCAE, Common Terminology Criteria for Adverse Events)
# _________________________
#    Acute
#    Dutz et al. 2019


# def NTCP_alopeciaGrade1(D2, beta_0=-0.94, beta_1=0.10):
#     # D2 == Skin D2%
#     return 1/(1+np.exp(-beta_0-beta_1*D2))


# Alopecia grade >=2 (CTCAE, Common Terminology Criteria for Adverse Events)
# _________________________
#    Acute
#    Dutz et al. 2019
# def NTCP_alopeciaGrade2(D5, beta_0=-1.33, beta_1=0.08):
#     # D5 == Skin D5%
#     return 1/(1+np.exp(-beta_0-beta_1*D5))


# Blindness
# _________________________
#    5 years post-RT
#    Burman et al. 1991
def NTCP_blindness(gEUD, TD50=65.0, m=0.14):
    # chiasm and optic nerves gEUD
    # gEUD = gEUD_calculator(d_i, v_i, a)
    t = (gEUD - TD50)/(m*TD50)
    # equals 1/sqrt(2pi) integral_-inf^t exp(-x^2/2)dx
    return 0.5*(1+scipy.special.erf(t/np.sqrt(2)))


# Brain necrosis
# _________________________
#    5 years post-RT
#    Bender et al. 2012
def NTCP_brainNecrosis(EQD2, D50=109.0, gamma=2.8, ab_ratio=0.96):
    # Brain-CTV and brain stem Dmax (EQD2)
    if EQD2 == 0:
        return 0
    else:
        return 1/(1+(D50/EQD2)**(4*gamma))


# Cataract requiring intervention
# _________________________
#    5 years post-RT
#    Burman et al. 1991
def NTCP_cataractRequiringIntervention(gEUD, TD50=18.0, m=0.27):
    # Lenses gEUD
    # gEUD = gEUD_calculator(d_i, v_i, a)
    t = (gEUD - TD50)/(m*TD50)
    # equals 1/sqrt(2pi) integral_-inf^t exp(-x^2/2)dx
    return 0.5*(1+scipy.special.erf(t/np.sqrt(2)))


# Delayed recall (on Wechsler Memory scale III Word Lists)
# _________________________
#    1.5 years post-RT
#    Gondi et al. 2012
# def NTCP_delayedRecall(EQD2, EQD_2_50=14.88, m=0.54, ab_ratio=2.0):
#     # Bilateral hippocampi D40% (EQD2)
#     t = (EQD2 - EQD_2_50)/(m*EQD_2_50)
#     # equals 1/sqrt(2pi) integral_-inf^t exp(-x^2/2)dx
#     return 0.5*(1+scipy.special.erf(t/np.sqrt(2)))


# Endocrine dysfunction (CTCAE, Common Terminology Criteria for Adverse Events)
# _________________________
#    At least 0.5 – 2 years post-RT
#    De Marzi et al. 2015
# def NTCP_endocrineDysfunction(gEUD, TD50=60.5, gamma50=5.2, a=6.4):
#     # Pituitary gEUD
#     # gEUD = gEUD_calculator(d_i, v_i, a)
#     if gEUD == 0:
#         return 0
#     else:
#         return 1/(1+(TD50/gEUD)**(4*gamma50))

# Erythema grade ≥ 1
# _________________________
#    Acute
#    Dutz et al. 2019


# def NTCP_erythema1(V35, beta_0=1.00, beta_1=0.09):
#     # Skin V35Gy(RBE), absolute volume
#     return 1/(1+np.exp(-beta_0-beta_1*V35))

# Erythema grade ≥ 2 (CTCAE, Common Terminology Criteria for Adverse Events)
# _________________________
#    Acute
#    Dutz et al. 2019


# def NTCP_erythema2(V35, beta_0=-1.54, beta_1=0.06):
#     # Skin V35Gy(RBE), absolute volume
#     return 1/(1+np.exp(-beta_0-beta_1*V35))


# Fatigue grade >=1 (CTCAE, Common Terminology Criteria for Adverse Events)
# _________________________
#    Acute
#    Dutz et al. 2019
def NTCP_fatigue1(D2, gender, beta_0=-0.90, beta_1=0.03, beta_2=1.28):
    # D2 == Brain-CTV(Gy) D2%
    # female: gender = 1
    # male:   gender = 0
    return 1/(1+np.exp(-beta_0-beta_1*D2-beta_2*gender))


# Hearing loss (CTCAE, Common Terminology Criteria for Adverse Events)
# _________________________
#    At least 0.5 – 2 years post-RT
#    De Marzi et al. 2015
def NTCP_hearingLoss(gEUD, TD50=56.0, gamma50=2.9, a=1.2):
    # Cochlea gEUD
    # gEUD = gEUD_calculator(d_i, v_i, a)
    if gEUD == 0:
        return 0
    else:
        return 1/(1+(TD50/gEUD)**(4*gamma50))


# Ocular toxicity grade ≥ 2 (RTOG, Radiation Therapy Oncology Group)
# _________________________
#    Acute
#    Batth et al. 2013
def NTCP_ocularToxicity(Dmax, beta_0=-5.174, beta_1=0.205):
    # Dmax == Ipsilateral lacrimal gland Dmax
    return 1/(1+np.exp(-beta_0-beta_1*Dmax))


# Temporal lobe injury
# _________________________
#    5 years post-RT
#    Kong et al. 2016
# def NTCP_temporalLobeInjury(Dmax, beta_0=-18.61, beta_1=0.227):
#     # Dmax == Temporal lobe D4max
#     return 1/(1+np.exp(-beta_0-beta_1*Dmax))


# Tinnitus grade ≥ 2 (LENT-SOMA, late effects of normal tissues - subjective, objective, management)
# _________________________
#    1–2 years post-RT
#    Lee et al. 2015
def NTCP_tinnitus2(Dmean, TD50=46.52, m=0.35):
    #  Cochlea Dmean
    t = (Dmean - TD50)/(m*TD50)
    # equals 1/sqrt(2pi) integral_-inf^t exp(-x^2/2)dx
    return 0.5*(1+scipy.special.erf(t/np.sqrt(2)))


# Late Tox

# Alopecia grade ≥1_12 months after PBT
# _________________________
#    Late
#    Dutz et al. 2021
# def NTCP_Alopecia_G1_12m__1(V45Gy, beta_0=-1.80, beta_1=0.15):
#     # Skin V45Gy(RBE) in cm^(-3)
#     return 1/(1+np.exp(-beta_0-beta_1*V45Gy))


# Alopecia grade ≥1_12 months after PBT
# _________________________
#    Late
#    Dutz et al. 2021
# def NTCP_Alopecia_G1_12m__2(D2, beta_0=-6.38, beta_1=0.15):
#     # D2 == Skin D2% in Gy(RBE)^-(1)
#     return 1/(1+np.exp(-beta_0-beta_1*D2))


# Alopecia grade ≥1_24 months after PBT
# _________________________
#    Late
#    Dutz et al. 2021
# def NTCP_Alopecia_G1_24m__1(V30Gy, beta_0=-1.70, beta_1=0.0048):
#     # Skin V30Gy(RBE) in cm^(-3)
#     return 1/(1+np.exp(-beta_0-beta_1*V30Gy))


# Alopecia grade ≥1_24 months after PBT
# _________________________
#    Late
#    Dutz et al. 2021
# def NTCP_Alopecia_G1_24m__2(D2, beta_0=-3.18, beta_1=0.068):
#     # D2 == Skin D2% in Gy(RBE)^-(1)
#     return 1/(1+np.exp(-beta_0-beta_1*D2))

# Hearing impairment grade ≥1_12 months after PBT
# _________________________
#    Late
#    Dutz et al. 2021


def NTCP_HearingImpairment_G1_12m__1(Dmean, beta_0=-3.03, beta_1=0.038):
    # Dmean == Cochlea ipsi Dmean in Gy(RBE)^-(1)
    return 1/(1+np.exp(-beta_0-beta_1*Dmean))


# Hearing impairment grade ≥1_12 months after PBT
# _________________________
#    Late
#    Dutz et al. 2021
def NTCP_HearingImpairment_G1_12m__2(Dmean, Age, beta_0=-7.02, beta_1=0.032, beta_2=0.072):
    # Dmean == Cochlea ipsi Dmean in Gy(RBE)^-(1)
    # Age == Age in years
    return 1/(1+np.exp(-beta_0-beta_1*Dmean-beta_2*Age))


# Hearing impairment grade ≥1_24 months after PBT
# _________________________
#    Late
#    Dutz et al. 2021
def NTCP_HearingImpairment_G1_24m(Dmean, beta_0=-3.48, beta_1=0.05):
    # Dmean == Cochlea ipsi Dmean in Gy(RBE)^-(1)
    return 1/(1+np.exp(-beta_0-beta_1*Dmean))


# Memory impairment grade ≥1_12 months after PBT
# _________________________
#    Late
#    Dutz et al. 2021
# def NTCP_MemoryImpairment_G1_12m(D2, beta_0=-2.32, beta_1=0.023):
#     # D2 == Hippocampi D2% in Gy(RBE)^-(1)
#     return 1/(1+np.exp(-beta_0-beta_1*D2))


# Memory impairment grade ≥1_24 months after PBT
# _________________________
#    Late
#    Dutz et al. 2021
def NTCP_MemoryImpairment_G1_24m(V35Gy, beta_0=-1.77, beta_1=6.5):
    # Brain-CTV V35Gy(RBE) as fraction of the total dose
    return 1/(1+np.exp(-beta_0-beta_1*V35Gy))


# Memory impairment grade ≥2_12 months after PBT
# _________________________
#    Late
#    Dutz et al. 2021
def NTCP_MemoryImpairment_G2_12m(V25Gy, beta_0=-3.42, beta_1=5.02):
    # Brain-CTV V25Gy(RBE) as fraction of the total dose
    return 1/(1+np.exp(-beta_0-beta_1*V25Gy))


# Fatigue grade ≥ 1_24 months after PBT
# _________________________
#    Late
#    Dutz et al. 2021
def NTCP_Fatigue_G1_24m(D2, CTx, beta_0=-1.52, beta_1=0.021, beta_2=-1.16):
    # D2 == BrainStem D2% in Gy(RBE)^-(1)
    # CTx == 0: patient recieved no chemotherapy
    # CTx == 1: patient recieved chemotherapy
    return 1/(1+np.exp(-beta_0-beta_1*D2-beta_2*CTx))


# Aus dem "dvh_rbe_functions.py" Packet aus Dortmund:
# get_EUD(path_rtdose, path_rtstructure, ROIname, parameter_a):

# def get_EQD2(path_rtdose, path_rtstructure, plan_path=None, dose_norm = 1,
#              CTgrid = False, path_ct_directory= None,settings = None):

# ToDiscuss for Monday:
# 1) Irgendwie müssen wir die "clinical goals" (Dmax, Dmean, D2, V35,...) und die richtigen ROIs zusammenbringen.
# 2) Und falls EQD2 oder EUD gebraucht wird, muss das berechnet werden (am besten aus den Input daten)
# -> inwiefern sollen wir das selber machen/ die Vorlage von Dortmund verwenden?! (@Martinas Meinung?!, da Dortmund Skript vor allem für die richtige Skalierung wichtig war.)
# 3) Input der "Klinischen Variablen" (age, gender, chemo,...)!

class LETtoRBEConverter():
    def __init__(self,
                 let_to_rbe_conversion,
                 alphabeta=2.,  # only used with 'wedenberg' mode
                 rbe_constant=1.1  # only used with 'constant' mode
                 ):
        assert let_to_rbe_conversion in ["wedenberg", "bahn", "constant"]
        self.let_to_rbe_conversion = let_to_rbe_conversion
        self.alphabeta = alphabeta
        self.rbe_constant = rbe_constant

    def __call__(self,
                 dose,
                 let,
                 dose_is_rbe_weighted,
                 clinical_var_df,
                 ):
        """
        Computes the RBE weighted dose from the given physical dose
        and a prediction of LET. The LET to RBE conversion is carried
        out using an established model from the literature.
        """
        assert dose.shape == let.shape
        print("Converting LET to RBE according to "
              f"{self.let_to_rbe_conversion.capitalize()}.")
        print("dose and let of shape", dose.shape)

        # we first convert the RBE-weighted dose to regular dose
        if dose_is_rbe_weighted:
            # convert to physical dose by division through 1.1
            physical_dose = dose / 1.1
            print("Dividing dose by 1.1 to obtain physical dose.")
        else:
            print("Dose is used as is, assuming to be physical dose!")
            physical_dose = dose

        if self.let_to_rbe_conversion == "wedenberg":
            # https://www.tandfonline.com/doi/full/10.3109/0284186X.2012.705892

            print(f"Alphabeta={self.alphabeta} will be used.")
            if clinical_var_df is None:
                raise ValueError(
                    "clinical_var_df is required as a pd.DataFrame but is None!")
            if "fxApplied" not in clinical_var_df:
                raise ValueError(
                    "Require column 'fxApplied' in clinical_var_df to compute "
                    "voxelwise dose per fraction in Wedenberg model."
                )

            n_fractions = int(clinical_var_df["fxApplied"])

            # voxelwise dose per fraction
            D = physical_dose / n_fractions
            # where D = 0, D_rec would be inf but we set it to zero.
            # This way, the RBEw for such voxels should be zero as well (even though in the Wedenberg model it is
            # kind of unclear how to proceed as the RBEw would be infinite)
            D_rec = 1. / D
            D_rec[D == 0] = 0

            q = 0.434

            # from https://www.tandfonline.com/doi/full/10.3109/0284186X.2012.705892, equation 3
            rbe = -0.5 * D_rec * self.alphabeta + D_rec * \
                np.sqrt(0.25 * self.alphabeta**2 +
                        (self.alphabeta + q * let) * D + D**2)

        elif self.let_to_rbe_conversion == "bahn":
            # https://www.sciencedirect.com/science/article/pii/S0360301620309366?via%3Dihub
            # equation (2) RBE = 1 + beta2/beta1 * LETd
            # with beta_1 = , beta_2 = as given in table 2
            beta2 = 0.018
            beta1 = 0.19
            rbe = 1. + (beta2 / beta1) * let

        elif self.let_to_rbe_conversion == "constant":
            # this assumes that LET has no effect on RBE
            # computation but is a fixed constant across
            # all dose voxels

            print(f"Using 'rbe_constant' {self.rbe_constant}.")
            rbe = self.rbe_constant * np.ones_like(let)
        else:
            raise ValueError(
                f"let_to_rbe_conversion {self.let_to_rbe_conversion} "
                "not understood.")

        # rbe weighted dose
        return rbe * physical_dose


class NTCPModelBase():
    def __init__(self,
                 impl_fn,
                 involved_roi):

        self.impl_fn = impl_fn
        self.involved_roi = involved_roi

    def compute_features(self, dose, roi_dict, clinical_var_df=None):
        assert dose.ndim == 3

        for k, v in roi_dict.items():
            if not v.shape == dose.shape:
                raise ValueError(
                    f"{k} has shape {v.shape}, but dose has shape {dose.shape}!")

        # we dont want to alter the entries in the roi_dict so we copy
        # and convert to boolean values
        our_roi_dict = {k: v.astype(bool) for k, v in roi_dict.items()}

        roi = self._create_roi(our_roi_dict)

        return self._compute_features(dose, roi, clinical_var_df)

    def _create_roi(self, roi_dict):
        raise NotImplementedError

    def _compute_features(self, dose, roi, clinical_var_df):
        raise NotImplementedError

    def __call__(self, dose, roi_dict, clinical_var_df=None):

        feature_dict = self.compute_features(
            dose, roi_dict, clinical_var_df)

        print("computed features are of shape")
        for k, v in feature_dict.items():
            print(k, v.shape)

        return self.impl_fn(**feature_dict)


class NTCPMemoryG2At12m(NTCPModelBase):
    def __init__(self):
        super().__init__(
            impl_fn=NTCP_MemoryImpairment_G2_12m,
            involved_roi="Brain-CTV"
        )

    def _create_roi(self, roi_dict):
        # ROI is Brain - CTV
        brain_mask = roi_dict["Mask_Brain"]
        ctv_mask = roi_dict["Mask_CTV"]
        # find the ROI: parts of the brain that are not in the CTV
        # i.e. all voxels where the brain_mask has a one but the ctv_mask has a zero
        # which should be all voxels that give us 1 when subtracting the ctv mask from the brain mask
        roi = (brain_mask.astype(int) - ctv_mask.astype(int)) == 1

        return roi

    def _compute_features(self, dose, roi, clinical_var_df):

        # get all dose values above the threshold within the roi
        # NOTE: in case we have batches, the sum should not include the first axis
        roi_high_dose = roi & (dose > 25)

        # NOTE: in here all arrays are 3D, i.e. Z x Y x X

        v25_frac = np.sum(roi_high_dose) / np.sum(roi)

        return {"V25Gy": v25_frac}


class NTCPMemoryG1At24m(NTCPModelBase):
    def __init__(self):
        super().__init__(
            impl_fn=NTCP_MemoryImpairment_G1_24m,
            involved_roi="Brain-CTV"
        )

    def _create_roi(self, roi_dict):
        # ROI is Brain - CTV
        brain_mask = roi_dict["Mask_Brain"]
        ctv_mask = roi_dict["Mask_CTV"]
        # find the ROI: parts of the brain that are not in the CTV
        # i.e. all voxels where the brain_mask has a one but the ctv_mask has a zero
        # which should be all voxels that give us 1 when subtracting the ctv mask from the brain mask
        roi = (brain_mask.astype(int) - ctv_mask.astype(int)) == 1

        return roi

    def _compute_features(self, dose, roi, clinical_var_df):

        roi_high_dose = roi & (dose > 35)
        # get all dose values above the threshold within the roi
        v35_frac = np.sum(roi_high_dose) / np.sum(roi)

        return {"V35Gy": v35_frac}


class NTCPFatigueG1At24m(NTCPModelBase):
    def __init__(self):
        super().__init__(
            impl_fn=NTCP_Fatigue_G1_24m,
            involved_roi="BrainStem"
        )

    def _create_roi(self, roi_dict):
        brainstem_mask = roi_dict["Mask_BrainStem"]
        return brainstem_mask

    def _compute_features(self, dose, roi, clinical_var_df):

        # NOTE: the keys have to match the parameters of the
        # self.impl_fn!

        assert "CTx" in clinical_var_df.columns
        ctx = clinical_var_df["CTx"].to_numpy()

        dose_at_brainstem = dose[roi]
        d2 = np.percentile(dose_at_brainstem, 98)

        return {
            "D2": d2,
            "CTx": ctx}


class NTCPBlindnessAt60mChiasm(NTCPModelBase):
    def __init__(self):
        super().__init__(
            impl_fn=NTCP_blindness,
            involved_roi="Chiasm"
        )

    def _create_roi(self, roi_dict):

        roi = roi_dict["Mask_Chiasm"]
        return roi

    def _compute_features(self, dose, roi, clinical_var_df):

        roi_dose = dose[roi]
        geud = gEUD_calculator(
            d_i=roi_dose,
            v_i=1/len(roi_dose),
            a=4)

        return {
            "gEUD": geud
        }


class NTCPBlindnessAt60mONBase(NTCPModelBase):
    def __init__(self, ipsilateral=True):
        ipsi_or_contra = "Ipsilateral" if ipsilateral else "Contralateral"
        self.ipsilateral = ipsilateral

        super().__init__(
            impl_fn=NTCP_blindness,
            involved_roi=f"{ipsi_or_contra}_OpticNerve"
        )

    def _create_roi(self, roi_dict):
        roi_left = roi_dict["Mask_OpticNerve_L"]
        roi_right = roi_dict["Mask_OpticNerve_R"]

        return (roi_left, roi_right)

    def _compute_features(self, dose, roi, clinical_var_df):
        roi_l, roi_r = roi

        # NOTE: decide on ipsilateral as the one with higher dose
        # print(i, dose.shape, glandl.shape, glandr.shape)
        l_dose_mean = dose[roi_l].mean()
        r_dose_mean = dose[roi_r].mean()

        if l_dose_mean >= r_dose_mean:
            # left is ipsilateral
            roi_ipsi = roi_l
            roi_contra = roi_r
        else:
            # right is ipsilateral
            roi_ipsi = roi_r
            roi_contra = roi_l

        # depending on whether our flag is on evaluating
        # ipsilateral or contralateral, we determine the
        # final roi from which we extract feature
        if self.ipsilateral:
            roi = roi_ipsi
        else:
            roi = roi_contra

        roi_dose = dose[roi]
        geud = gEUD_calculator(
            d_i=roi_dose,
            v_i=1/len(roi_dose),
            a=4)

        return {
            "gEUD": geud
        }


class NTCPBlindnessAt60mONIpsi(NTCPBlindnessAt60mONBase):
    def __init__(self):
        super().__init__(ipsilateral=True)


class NTCPBlindnessAt60mONContra(NTCPBlindnessAt60mONBase):
    def __init__(self):
        super().__init__(ipsilateral=False)


class NTCPBrainNecrosisAt60m(NTCPModelBase):
    def __init__(self):
        super().__init__(
            impl_fn=NTCP_brainNecrosis,
            involved_roi="Brain-CTV"
        )

    def _create_roi(self, roi_dict):
        brain_mask = roi_dict["Mask_Brain"]
        ctv_mask = roi_dict["Mask_CTV"]
        # find the ROI: parts of the brain that are not in the CTV
        # i.e. all voxels where the brain_mask has a one but the ctv_mask has a zero
        # which should be all voxels that give us 1 when subtracting the ctv mask from the brain mask
        roi = (brain_mask.astype(int) - ctv_mask.astype(int)) == 1

        return roi

    def _compute_features(self,
                          dose,
                          roi,
                          clinical_var_df):

        # Brain-CTV and brain stem Dmax (EQD2)
        raise NotImplementedError

        return {
            "EQD2": None,
        }


class NTCPCataractAt60mBase(NTCPModelBase):
    def __init__(self, mode):
        assert mode in ["ipsilateral", "contralateral", "merged"]
        self.mode = mode

        super().__init__(
            impl_fn=NTCP_cataractRequiringIntervention,
            involved_roi="Lenses"
        )

    def _create_roi(self, roi_dict):
        # Lenses gEUD
        lensl_mask = roi_dict["Mask_Lens_L"]
        lensr_mask = roi_dict["Mask_Lens_R"]

        return (lensl_mask, lensr_mask)

    def _compute_features(self,
                          dose,
                          roi,
                          clinical_var_df):

        roi_l, roi_r = roi
        if self.mode == "merged":
            roi = fuse_rois([roi_l, roi_r])
        else:
            # we got to determine ipsi and contralateral
            l_dose_mean = dose[roi_l].mean()
            r_dose_mean = dose[roi_r].mean()

            if l_dose_mean >= r_dose_mean:
                # left is ipsilateral
                roi_ipsi = roi_l
                roi_contra = roi_r
            else:
                # right is ipsilateral
                roi_ipsi = roi_r
                roi_contra = roi_l

            if self.mode == "ipsilateral":
                roi = roi_ipsi
            else:
                roi = roi_contra

        roi_dose = dose[roi]
        geud = gEUD_calculator(
            d_i=roi_dose,
            v_i=1/len(roi_dose),
            a=3.33)

        return {
            "gEUD": geud,
        }


class NTCPCataractAt60mMerged(NTCPCataractAt60mBase):
    def __init__(self):
        super().__init__(
            mode="merged"
        )


class NTCPCataractAt60mIpsi(NTCPCataractAt60mBase):
    def __init__(self):
        super().__init__(
            mode="ipsilateral"
        )


class NTCPCataractAt60mContra(NTCPCataractAt60mBase):
    def __init__(self):
        super().__init__(
            mode="contralateral"
        )


class NTCPFatigueG1Acute(NTCPModelBase):
    def __init__(self):
        super().__init__(
            impl_fn=NTCP_fatigue1,
            involved_roi="Brain-CTV"
        )

    def _compute_features(self,
                          dose,
                          roi,
                          clinical_var_df):
        # D2 == Brain-CTV(Gy) D2%
        # female: gender = 1
        # male:   gender = 0
        raise NotImplementedError
        return {
            "D2": None,
            "gender": None,
        }


class NTCPHearingLossAt24m(NTCPModelBase):
    def __init__(self):
        super().__init__(
            impl_fn=NTCP_hearingLoss,
            involved_roi="Cochlea"
        )

    def _compute_features(self,
                          dose,
                          roi,
                          clinical_var_df):
        # Cochlea gEUD
        # NOTE: we dont have cochlea, maybe use Ear_internal_L or Ear_internal_R
        raise NotImplementedError
        return {
            "gEUD": None
        }


class NTCPOcularToxicityG2AcuteBase(NTCPModelBase):
    def __init__(self, ipsilateral=True):
        ipsi_or_contra = "Ipsilateral" if ipsilateral else "Contralateral"
        self.ipsilateral = ipsilateral

        super().__init__(
            impl_fn=NTCP_ocularToxicity,
            involved_roi=f"{ipsi_or_contra}_LacrimalGland"
        )

    def _create_roi(self, roi_dict):
        # Dmax == Ipsilateral lacrimal gland Dmax
        glandl_mask = roi_dict["Mask_LacrimalGland_L"]
        glandr_mask = roi_dict["Mask_LacrimalGland_R"]
        # print("mask dtype", glandl_mask.dtype, glandl_mask.shape)

        return (glandl_mask, glandr_mask)

    def _compute_features(self, dose, roi, clinical_var_df):
        glandl, glandr = roi

        glandl_dose_mean = dose[glandl].mean()
        glandr_dose_mean = dose[glandr].mean()

        if glandl_dose_mean >= glandr_dose_mean:
            # left is ipsilateral
            roi_ipsi = glandl
            roi_contra = glandr
        else:
            # right is ipsilateral
            roi_ipsi = glandr
            roi_contra = glandr

        if self.ipsilateral:
            roi = roi_ipsi
        else:
            roi = roi_contra

        dmax = dose[roi].max()

        return {
            "Dmax": dmax
        }


class NTCPOcularToxicityG2AcuteIpsi(NTCPOcularToxicityG2AcuteBase):
    def __init__(self):
        super().__init__(ipsilateral=True)


class NTCPOcularToxicityG2AcuteContra(NTCPOcularToxicityG2AcuteBase):
    def __init__(self):
        super().__init__(ipsilateral=False)


class NTCPTinnitusG2At24m(NTCPModelBase):
    def __init__(self):
        super().__init__(
            impl_fn=NTCP_tinnitus2,
            involved_roi="Cochlea"
        )

    def _compute_features(self,
                          dose,
                          roi,
                          clinical_var_df):
        #  Cochlea Dmean
        # NOTE: we dont have cochlea, maybe use Ear_internal_L or Ear_internal_R
        raise NotImplementedError
        return {
            "Dmean": None,
        }


class NTCPHearingImpairmentG1At12m(NTCPModelBase):
    def __init__(self):
        super().__init__(
            impl_fn=NTCP_HearingImpairment_G1_12m__1,
            involved_roi="Cochlea"
        )

    def _compute_features(self,
                          dose,
                          roi,
                          clinical_var_df):

        # Dmean == Cochlea ipsi Dmean in Gy(RBE)^-(1)
        # NOTE: we dont have cochlea, maybe use Ear_internal_L or Ear_internal_R
        raise NotImplementedError
        return {
            "Dmean": None,
        }


class NTCPHearingImpairmentG1At12mWithAge(NTCPModelBase):
    def __init__(self):
        super().__init__(
            impl_fn=NTCP_HearingImpairment_G1_12m__2,
            involved_roi="Cochlea"
        )

    def _compute_features(self,
                          dose,
                          roi,
                          clinical_var_df):

        # Dmean == Cochlea ipsi Dmean in Gy(RBE)^-(1)
        # NOTE: we dont have cochlea, maybe use Ear_internal_L or Ear_internal_R
        # Age == Age in years
        raise NotImplementedError
        return {
            "Dmean": None,
            "Age": None,
        }


class NTCPHearingImpairmentG1At24m(NTCPModelBase):
    def __init__(self):
        super().__init__(
            impl_fn=NTCP_HearingImpairment_G1_24m,
            involved_roi="Cochlea"
        )

    def _compute_features(self,
                          dose,
                          roi,
                          clinical_var_df):

        # Dmean == Cochlea ipsi Dmean in Gy(RBE)^-(1)
        # NOTE: we dont have cochlea, maybe use Ear_internal_L or Ear_internal_R
        raise NotImplementedError
        return {
            "Dmean": None,
        }
