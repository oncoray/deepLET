# -*- coding: utf-8 -*-
"""
Input variables

by Aaron Kieslich, 19.01.2023
"""

# table with DICOM paths
table_path = "../../../data/tables/Plan_table.xlsx"
table_sheet ="Waterphantom" 

# table to rename ROIs
table_ROI_path = "../../../data/tables/ROInaming_table.xlsx"

# path to folder with dcm data
path_dcm_data = "../../../data/waterphantom/dcm"

# path to csv file with bad patients
path_bad_plans = "../../../data/tables/bad_patients.csv"

# path to save on
path_save_interpolated_data = "../../../data/waterphantom/npy"
path_save_check_figures = "../../../data/figures"

# new voxel spacing
new_voxel_spacing = (1,1,1)

# variable, which assigns, whether just the patients in path_bad_plans should
# be analysed, or all patients
analyse_bad_plans = False

# list of manually defined plans to analyse
# overwrites analyse_bad_plans option
analyze_plans = []

# set rois which should be extracted
rois_to_extract = ["External", "Brain", "BrainStem", "CTV"]
