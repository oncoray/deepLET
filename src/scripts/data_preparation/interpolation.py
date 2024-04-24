# -*- coding: utf-8 -*-
"""
Created 19.01.2023

@author: Aaron Kieslich + Dortmund

Maps the MC-dose, LETt, LETd and ROIs on the CT and then interpolates
everything to an equal voxel size.
Inputs can be specified on input_variables.py.
Following convention for the dimensions of the CT image is done:
x refers to the sagital plane (column).
y refers to the coronal plane (row).
z refers to the transveral plane (depth).

"""

import os 
import numpy as np # Version 1.20.2
import pydicom # Version 1.3.0
from pymedphys import dicom # Version 0.36.1
from dicompylercore import dicomparser # Version 0.5.5
from scipy.interpolate import RegularGridInterpolator as rgi # Version 1.6.2
from Levenshtein import distance as levenshtein_distance

from scipy.ndimage import zoom

from matplotlib.path import Path
from matplotlib import pyplot as plt

import pandas as pd

# get input variables
from input_variables import table_path, table_ROI_path, table_sheet , \
    path_save_interpolated_data, path_save_check_figures, new_voxel_spacing, path_bad_plans, \
    analyse_bad_plans, analyze_plans, rois_to_extract, path_dcm_data
#from bad_plans import bad_plans

from tqdm import tqdm

# prevents error
pydicom.config.settings.reading_validation_mode = "WARN"


def get_ct_files(ct_directory):
    '''catches every CT slice from folder'''
    ct_data = []
  
    # Durchsucht den Auswertungsordner nach den entsprechenden Dateien
    for subdir, dirs, files in os.walk(ct_directory):
         # Zwischenspeicher für CT-Dateien
        for file in files:
            data_path = os.path.join(subdir, file)
            data = dicomparser.DicomParser(data_path)
            modality = data.ds.Modality
            if modality == 'CT':
                ct_data.append(os.path.join(subdir, file))

    if len(ct_data) == 0:
        raise Exception('Fehlende CT-Dateien') 
           
    return ct_data


def get_ct_grid(ct_directory):
    '''
    Calculating Arrays for x,y,z with location of CT Voxel in mm.
    x refers to the sagital plane (column).
    y refers to the coronal plane (row).
    z refers to the transveral plane (depth).
    '''
    ct_files = get_ct_files(ct_directory)
    
    location_tupel = np.zeros(len(ct_files))
    # Bestimmt die Schichtlage jedes CT-Bildes und speichert sie zusammen
    # mit der entsprechenden CT-Datei ab
    i = 0
    for ct_file in ct_files:
        ct_data = dicomparser.DicomParser(ct_file)
        ct_location = ct_data.ds.SliceLocation
        location_tupel[i] = ct_location
        i +=1
    
    #get CT slice data 
    CTslice = dicomparser.DicomParser(ct_files[0])
    
    CTimgpos            = CTslice.ds.ImagePositionPatient
    CTpixelspacing      = CTslice.ds.PixelSpacing
    CTrows              = CTslice.ds.Rows
    CTcolumns           = CTslice.ds.Columns
        
    # CT-Koordinaten bestimmen
    x_CT = np.arange(CTcolumns)    *   CTpixelspacing[0] + CTimgpos[0]
    y_CT = np.arange(CTrows)       *   CTpixelspacing[1] + CTimgpos[1]
    
    #CT z-coordinates from z-locations of existing slices
    z_CT = sorted(location_tupel)

    return np.array(z_CT), y_CT, x_CT


def get_ROImask_on_ctgrid_interpolate(ct_directory, path_rtstructure, \
                                      ROInumber, path_rtdose,  \
                                          binary_mask = True, threshhold=0.5): 
    
    '''Calculate binary mask for ROI on CT grid. Written by Dortmund.
    
    Parameters
    ------------------
    ct_directory: path to directory with CT Dicom files (string)
    path_rtstructure: path to RT Structure Set Dicom (string)
    ROInumber: Number of ROI to interpolate from Structure set (int)
    path_rtdose: path to RT Plan file Dicom (string
    binary_mask = True: Retunr is a  t3D array with binary inputvoxels inside 
            ROI =1, outside =0
    threshhold=0.5: value where above interpolated voxels are counted in ROI 

    Returns
    ------------------
    struc_dosegrid_interp: 3D array with ROI mask, size of CT grid cutted down
                            to extents of dosegrid in x and y direction 
    '''
    #get dose data
    rtdose_obj = dicomparser.DicomParser(path_rtdose)   
    dosedata   = rtdose_obj.GetDoseData() 
    
    #get extents of dosegrid in x and y direction
    x_dose_min = np.min(dosedata['lut'][0])
    x_dose_max = np.max(dosedata['lut'][0])
    y_dose_min = np.min(dosedata['lut'][1])
    y_dose_max = np.max(dosedata['lut'][1])
    
    z_CT, y_CT, x_CT = get_ct_grid(ct_directory) #arrays with ct voxel coordinates
        
    #cut ct coordinates to grid in dosegrid 
    x_index = np.where((x_CT >= x_dose_min-2)*(x_CT <= x_dose_max+2))
    y_index = np.where((y_CT >= y_dose_min-2)*(y_CT <= y_dose_max+2))
    
    x_CT_cut = x_CT[x_index]
    y_CT_cut = y_CT[y_index]
    
    #fill coordinates in 2d array
    x, y = np.meshgrid(x_CT, y_CT)
    x, y = x.flatten(), y.flatten()
    ctgridpoints = np.vstack((x, y)).T
    
    #get coordinates for original structure contours
    rtstructure_file = dicomparser.DicomParser(path_rtstructure)
    ROIcoords_orig  = (rtstructure_file.GetStructureCoordinates(ROInumber))

    ROIcoords_z = np.array(list(ROIcoords_orig.keys())).astype(float)
    ROI_extents_zmin, ROI_extents_zmax = np.amin(ROIcoords_z), np.amax(ROIcoords_z)    

    keys = list(ROIcoords_orig.keys()) #list of every z-slice coordinate where
    #ROI has a contour
    
    roimask_3d = np.zeros(( len(y_CT),len(x_CT), len(keys) ))
    i = 0
    for key in keys:  #going through all z slices
      
        j = 0
        gridslice = np.zeros( ( len(y_CT),len(x_CT) ) )
        
        #going through all contours in one slice
        for j in range(len( ROIcoords_orig[key])): 
            #create contour of ROI coordinates in slice
            ROI_datapoints = np.array(ROIcoords_orig[key][j]['data'])
            contour =  ROI_datapoints[:,0:2]
            c = Path(list(contour))

            #check if points of ct grid are inside or outside of contour
            grid = c.contains_points(ctgridpoints) 
            
            #stack filled contours and holes
            gridslice = gridslice + grid.reshape(( len(y_CT), len(x_CT) )) 
            
            j+=1
            
        #use modulo to get stacked contours to holes and structure again
        gridslice = gridslice % 2 
        
        #save contour in slice in 3d array
        roimask_3d[:,:,i] = gridslice
        i+=1
            
    
    #get structure coordinate in z axis
    zcoax = ROIcoords_z
    
    #Interpolation des gesamten Dosisgrid in der Maske der Struktur (auf Dosisgrid)

    #CT grid as reference grid (in x,y dimensions cutted to extents of dose grid)
    #Refernce grid here: points which are interpolated and retunred
    ref_coords_z = z_CT[(z_CT <= ROI_extents_zmax+2) * (z_CT >= ROI_extents_zmin-2)]
    ref_coords_rows = y_CT_cut
    ref_coords_columns = x_CT_cut

    #3D Grid where Structures have been conoured and interpolated in z-slices 
    eval_coords_z = zcoax
    eval_coords_rows = y_CT
    eval_coords_columns = x_CT

    eval_coords = (eval_coords_z, eval_coords_rows, eval_coords_columns)

    roimask = np.zeros((len(roimask_3d[0,0,:]), len(roimask_3d[:,0,0]), len(roimask_3d[0,:,0])))

    #change ROI mask in zyx system
    i = 0
    for i in range(len(roimask_3d[0,0,:])): 
        roimask[i,:,:] = roimask_3d[:,:,i]
        i += 1
    
    eval_dose = roimask
    
    # Interpolationfunction filled with coordinates and binary mask per zslice
    my_interpolating_function = rgi(eval_coords, eval_dose, method = 'linear', 
                                    bounds_error=False, fill_value=0)


    # Create Refernce grid from x and y values
    yarr, xarr = np.meshgrid(ref_coords_rows, ref_coords_columns)
    yarr = np.ravel(yarr)
    xarr = np.ravel(xarr)
    z_ones = np.ones_like(xarr)
    
    # Index-Array
    yindarr, xindarr = np.meshgrid(y_index, x_index)
    yindarr = np.ravel(yindarr)
    xindarr = np.ravel(xindarr)
    
    # Create new structure array in shape of ct grid
    evaldose_on_refgrid = np.zeros((len(z_CT), len(y_CT), len(x_CT)))
    
    zgo = 0 #Runindex for index in z-Direction 
    
    # Interpolation slicewise
    for sliceloc in ref_coords_z:
        # Koordinate für korrekte Interpolation
        zarr = z_ones * sliceloc
        zgo = np.where(z_CT==sliceloc)
        if sliceloc in eval_coords_z :
            #No interpolation, if slice is at location of original 
            #interpolated slices
            zgoarr = np.array(z_ones, dtype=int) * zgo
            inx_z_stru = np.where(eval_coords_z == sliceloc)
            evaldose_on_refgrid[zgo, :, :] = eval_dose[inx_z_stru[0],:,:]
            
        else : 
            # Interpoliert das Auswertungsarray auf die 
            # Koordianten des Referenzarrays
            intpolarr = my_interpolating_function(np.array([zarr, yarr, xarr]).T)
            # Index-Array für korrekte Eintragung in das neue Array
            zgoarr = np.array(z_ones, dtype=int) * zgo
            # Eintragen der interpolierten Werte in das neue Array
            evaldose_on_refgrid[zgoarr, yindarr, xindarr] = intpolarr
        
    #set values over threshhold to one and beneith to 0
    if binary_mask == True: 
        struc_ctgrid_interp = np.zeros(evaldose_on_refgrid.shape)
        struc_ctgrid_interp[evaldose_on_refgrid>threshhold] = 1
    else: 
         struc_ctgrid_interp = evaldose_on_refgrid
     
    return struc_ctgrid_interp

 
def get_dose_on_ctgrid(ct_directory, path_dose): 
    '''Calculate Dose values in 3D Array on Voxels of CT Grid by linear interpolation
    ----------
    path_dose: string
        File path to RT Dose file
    ct_directory: string 
        File path to directory holding all CT images 
    
    Returns
    ----------
        3D Numpy Array, voxel resolution of CT images, cutted to size of dose 
        grid in x and y dimension, not cutted in z dimension 
    '''

    # get CT grid
    z_CT, y_CT, x_CT = get_ct_grid(ct_directory)
    ref_coords_z = z_CT
    ref_coords_rows = y_CT
    ref_coords_columns = x_CT
    
    # get dose data
    eval_coords = get_dose_grid(path_dose)
    eval_dose = get_dose_array(path_dose)

    # Interpolationsfunktion
    my_interpolating_function = rgi(eval_coords, eval_dose, method = 'linear', 
                                    bounds_error=False, fill_value=0)

    # Erstellen eines Referenz-Gitters aus x und y Werten
    yarr, xarr = np.meshgrid(ref_coords_rows, ref_coords_columns)
    yarr = np.ravel(yarr)
    xarr = np.ravel(xarr)
    z_ones = np.ones_like(xarr)
    
    # Index-Array
    yindarr, xindarr = np.meshgrid(np.arange(len(ref_coords_rows)),
                                   np.arange(len(ref_coords_columns)))
    yindarr = np.ravel(yindarr)
    xindarr = np.ravel(xindarr)
    
    # Erstellen des neuen Dosis-Arrays in der entsprechenden Größe des CTgrids
    evaldose_on_refgrid = np.zeros((len(z_CT), len(y_CT), len(x_CT)))
    
    zgo = 0 # Laufvariable für Indexwerte in z-Richtung
    
    # Schichtweise Interpolation
    for sliceloc in ref_coords_z:
        
        # Koordinate für korrekte Interpolation
        zarr = z_ones * sliceloc
        
        intpolarr = my_interpolating_function(np.array([zarr, yarr, xarr]).T)
        
        # Index-Array für korrekte Eintragung in das neue Array
        zgoarr = np.array(z_ones, dtype=int) * zgo
        
        # Eintragen der interpolierten Werte in das neue Array
        evaldose_on_refgrid[zgoarr, yindarr, xindarr] = intpolarr
        
        zgo += 1
     
    return evaldose_on_refgrid


def get_ct_voxel_spacing(path_ct):
    """
    Returns the Voxel Spacing. Assumes that every file in folder has the same
    Voxel spacing.

    Parameters
    ----------
    path : to folder containing dcm-file

    """
    
    voxel_size_x = []
    voxel_size_y = []
    voxel_size_z = []
    
    ct_files = get_ct_files(path_ct)
    
    # loop through all the DICOM files in the directory
    for filename in ct_files:
        filepath = os.path.join(path_ct, filename)
        # read the DICOM file
        ds = pydicom.dcmread(filepath, force = True)
        
        # extract the voxel size from the PixelSpacing and SliceThickness attribute
        voxel_size_x.append(ds.PixelSpacing[0])
        voxel_size_y.append(ds.PixelSpacing[1])
        voxel_size_z.append(float(ds.SliceThickness))

    voxel_size_x = np.unique(voxel_size_x)
    voxel_size_y = np.unique(voxel_size_y)
    voxel_size_z = np.unique(voxel_size_z)
        
    if len(voxel_size_x) != 1 or len(voxel_size_y) != 1 or len(voxel_size_z) != 1:
        raise ValueError("Voxel sizes are not unique!")
        
    voxel_size = (voxel_size_z[0], voxel_size_y[0], voxel_size_x[0])

    return voxel_size


def extract_sliceloc(filename):
    """
    Extracts the slice location of a .dcm file. 
    """
    
    ds = pydicom.dcmread(filename, force = True)
    
    return ds.SliceLocation
    

def get_ct_array(path_CT):
    """
    Returns 3D array containing the CT values in HU
    """
    
    # specify the directory containing the DICOM files
    dirpath = path_CT

    # initialize an empty list to store the CT data
    ct_data = []
    
    list_files = get_ct_files(path_CT)
    
    # get sorted file list by the file name.
    sorted_list = sorted(list_files, key=lambda x: extract_sliceloc(x))
    
    
    # loop through all the DICOM files in the directory
    for filename in sorted_list:

        # read the DICOM file
        ds = pydicom.dcmread(os.path.join(dirpath, filename), force = True)
        
        pixel_data = np.array(ds.pixel_array)
        
        # translate pixel array value in HU
        intercept = ds.RescaleIntercept
        slope = ds.RescaleSlope
        hu_values = pixel_data * slope + intercept
        
        # add the CT data to the list
        ct_data.append(hu_values)
    
    
    ct_data = np.array(ct_data, dtype = np.float32)
    
    return ct_data


def change_voxel_spacing(array, old_voxel_spacing, new_voxel_spacing = (1,1,1)):
    """
    Changes the Voxel size of an array to a new Voxel size by linear interpolation.
    x refers to the sagital plane (column).
    y refers to the coronal plane (row).
    z refers to the transveral plane (depth).
    
    Parameters
    ----------------
    old_voxel_spacing: The currenct voxel spacing of the array (in mm). (z,y,x).
    new_voxel_spacing; New wanted voxel spacing of the interpolated array (in mm). (z,y,x).
    """
    
    # calculate the zoom factor for each dimension
    zoom_factor = np.array([old_voxel_spacing[0]/new_voxel_spacing[0], \
                            old_voxel_spacing[1]/new_voxel_spacing[1], \
                                old_voxel_spacing[2]/new_voxel_spacing[2]])

    # interpolate the array using zoom function
    array = zoom(array, zoom_factor, mode='constant', cval = np.min(array), \
                 order = 1)
    
    return array.astype(np.float32)



def get_dose_array(path_dose):
    """
    Returns 3D array with Dose data.
    """
    ds = pydicom.dcmread(path_dose, force = True)

    # get dose data 
    grid, dose_data = dicom.zyx_and_dose_from_dataset(ds)
    
    return dose_data


def get_dose_grid(path_dose):
    '''
    Calculating Arrays for x,y,z with location of CT Voxel in mm.
    x refers to the sagital plane (column).
    y refers to the coronal plane (row).
    z refers to the transveral plane (depth).
    '''
    
    ds = pydicom.dcmread(path_dose, force = True)
    
    #get dose data 
    grid, value = dicom.zyx_and_dose_from_dataset(ds)

    z_dose = grid[0]
    y_dose = grid[1]
    x_dose = grid[2]
    
    return z_dose, y_dose, x_dose 


def update_bad_plans_list(bad_plans, good_plans, file_name):
    """
    Add all plans with error (bad_plans) to csv-file. Create file
    if not existing. Exclude all plans without error (good_plans) from 
    file.
    """
    
    if not os.path.exists(file_name):
        df = pd.DataFrame({"plan_id":[]})
    else:    
        try:
            df = pd.read_csv(file_name)
        except pd.errors.EmptyDataError:
            print('File is empty')
            df = pd.DataFrame({"plan_id":[]})
    
    # update bad plan list
    new_bad_plans = np.concatenate((df["plan_id"].values, np.array(bad_plans)))
    new_bad_plans = np.unique(new_bad_plans)
    
    # exclude good plans
    new_bad_plans = new_bad_plans[~np.isin(new_bad_plans, good_plans)]
    
    # save new list
    df_new = pd.DataFrame({"plan_id":new_bad_plans})
    df_new.to_csv(file_name, index = False)
    
    return


def find_matching_roi(roi_list, target_roi, df_mapping):
    """
    Finds the matching ROI name in a list of potentially fussy and strange ROI names, adapted to work with a restructured DataFrame.

    Parameters:
    roi_list (list): List of existing ROI names.
    target_roi (str): The ROI name to be extracted.
    df_mapping (DataFrame): DataFrame containing the mapping between old ROI names (as rows) and new ROI names (as columns).

    Returns:
    str: The matching ROI name from roi_list, or None if no match is found.
    """

    # Step 1: Check if the exact ROI name is in the list
    if target_roi in roi_list:
        return target_roi

    # Step 2: Check for ROI names that match with the target ROI in the restructured DataFrame
    if target_roi in df_mapping.columns:
        potential_matches = df_mapping[target_roi].dropna().tolist()
        for potential_match in potential_matches:
            if potential_match in roi_list:
                return potential_match

    # Step 3: Iterate through the fussy ROI list to find the closest match using Levenshtein distance
    for fussy_roi in roi_list:
        
        # Find the ROIname that has the minimal Levenshtein distance to the fussy_roi
        closest_roi_new = None
        min_distance = float('inf')
        
        for col in df_mapping.columns:
            df_mapping['Levenshtein_Distance_to_Fussy'] = df_mapping[col].apply(lambda x: levenshtein_distance(str(x), fussy_roi) if pd.notna(x) else float('inf'))
            current_min_distance = df_mapping['Levenshtein_Distance_to_Fussy'].min()
            
            if current_min_distance < min_distance:
                min_distance = current_min_distance
                closest_roi_new = col
                
        # Check if the renamed ROI matches the target ROI
        # difference between the words shouldn't be too high
        if closest_roi_new == target_roi and min_distance <= len(fussy_roi) / 2:
            return fussy_roi
            
    return None

if __name__ == "__main__":

    # table with DICOM paths
    table = pd.read_excel(table_path, sheet_name=table_sheet)
    
    # table to rename ROIs
    table_ROI = pd.read_excel(table_ROI_path)
    
    # create save folder, if not existing
    if not os.path.exists(path_save_check_figures):
        os.makedirs(path_save_check_figures)
    
    # check if manually defined plans list is empty
    # if not empty, analyse this plans
    if len(analyze_plans) != 0:
        
        # just take the plans, which are also defined in table with DICOM paths
        plan_list = np.intersect1d(np.array(analyze_plans),table["plan_id"].values)
        print("Analysing manually defined plans: ", plan_list)
    
    # just analyse bad plans
    elif analyse_bad_plans == True:
        assert os.path.exists(path_bad_plans) == True, "Create bad_plans.csv or set analyse_bad_plans to False."
        
        df = pd.read_csv(path_bad_plans)
        
        # just take the plans, which are also defined in table with DICOM paths
        plan_list = np.intersect1d(df["plan_id"].values,table["plan_id"].values)
        
        not_analyzed_plans = np.setdiff1d(df["plan_id"].values,table["plan_id"].values)
        
        print("Bad plans, which are not in DICOM-table: ", not_analyzed_plans)
        print("Analysing bad plans: ", plan_list)
    else:
        print("Analysing all plans in DICOM-path table!")
        plan_list = table["plan_id"].values
    
    # list for plans, where errors occured
    bad_plans_save = []
    
    # list for plans, where no errors occured
    good_plans_save = []
    
    # check if plans list is not empty
    assert len(plan_list) != 0
    
    for plan in tqdm(plan_list):
        try:
            
            print("plan: " + str(plan))
            
            # get paths from excel-table
                        
            path_dose = os.path.join(path_dcm_data, plan, "plan_dose.dcm")
            path_LETd = os.path.join(path_dcm_data, plan, "LETd.dcm")
            path_RTSS = os.path.join(path_dcm_data, plan, "RTSS.dcm")
            path_CT = os.path.join(path_dcm_data, plan, "CT")
         
            # Assertions to check if paths exist
            assert os.path.exists(path_dose), f"The path for planned dose does not exist: {path_dose}"
            assert os.path.exists(path_LETd), f"The path for LETd does not exist: {path_LETd}"
            assert os.path.exists(path_RTSS), f"The path for RTSS does not exist: {path_RTSS}"
            assert os.path.exists(path_CT), f"The path for CT directory does not exist: {path_CT}"
             
            # get CT array
            array_CT = get_ct_array(path_CT).astype(np.float32)
            ct_voxel_spacing = get_ct_voxel_spacing(path_CT)
            
            # read RTSS
            ds = pydicom.dcmread(path_RTSS, force = True)
            
            list_dcm_roi_names = []
            
            # iterate over the StructureSetROISequence
            for roi in ds.StructureSetROISequence:
                
                # extract the ROI name and roi_number
                list_dcm_roi_names.append(roi.ROIName)
            
            print(f"All ROI names in RTSS: {list_dcm_roi_names}")
            roi_result = {}
            
            print("Extracting ROIs....")
            
            for roi_name_to_extract in rois_to_extract:
                if roi_name_to_extract == "CTV":
                    matching_roi = table.loc[table["plan_id"] == plan, roi_name_to_extract].values[0]
                elif roi_name_to_extract == "CTV_SiB":
                    if pd.isna(table.loc[table["plan_id"] == plan, "CTV_SiB"].values[0]) == False:
                        matching_roi = table.loc[table["plan_id"] == plan, roi_name_to_extract].values[0]
                    else:
                        print(f"Plan {plan} has no SiB.")
                        continue
                else:
                    matching_roi = find_matching_roi(list_dcm_roi_names, roi_name_to_extract, table_ROI)
                if matching_roi == None:
                    print(f"{roi_name_to_extract} not found. Skip ROI!")
                    continue
                
                for roi in ds.StructureSetROISequence:
                    if roi.ROIName == matching_roi:
                        roinumber = roi.ROINumber
                
                print(f"Extract {roi_name_to_extract}. DCM name: {matching_roi}")
                try:
                    # get ROI mask
                    rois = get_ROImask_on_ctgrid_interpolate(path_CT, path_RTSS, \
                                                              roinumber, path_dose,\
                                                              binary_mask = True, \
                                                              threshhold=0.5)
                except Exception as e:
                    print(e)
                    print(f"Interpolation for {roi_name_to_extract} failed.")
                    
                roi_1mm = change_voxel_spacing(rois, ct_voxel_spacing, new_voxel_spacing)
                roi_1mm[roi_1mm <= 0.5] = 0
                roi_1mm[roi_1mm > 0.5] = 1
                
                if np.sum(roi_1mm) == 0:
                    print(f"{roi_name_to_extract} isn't located on CT!")
                    continue
                
                # get the ROI 'External' to check for right positioning of 
                # ROIs on the CT
                # The External (outer contour) represents the area in which 
                # the dose is calculated. It usually includes the plan
                # but no surrounding air.
                if roi_name_to_extract == "External":
                    check_ROI = roi_1mm.astype(bool)
                
                # save result
                roi_result[roi_name_to_extract] = roi_1mm.astype(bool)
            
            assert len(roi_result) > 0, "No ROIs extracted."
                
            # create save folder, if not existing
            if not os.path.exists(os.path.join(path_save_interpolated_data,plan)):
                os.makedirs(os.path.join(path_save_interpolated_data,plan))
            
            # save ROIs
            np.save(os.path.join(path_save_interpolated_data,plan, "ROIs"), \
                    roi_result)
            
            print("ROIs saved!")
            
            # delete because of memory
            del roi_result
        
            print("Interpolating doses...")
            dose_interpol_on_ct = get_dose_on_ctgrid(path_CT, path_dose)
            dose_intpol_on_new_size = change_voxel_spacing(dose_interpol_on_ct, ct_voxel_spacing, new_voxel_spacing)
            
            assert np.sum(dose_intpol_on_new_size) > 0, "No dose values on CT!"
            
            LETd_interpol_on_ct = get_dose_on_ctgrid(path_CT, path_LETd)
            LETd_intpol_on_new_size = change_voxel_spacing(LETd_interpol_on_ct, ct_voxel_spacing, new_voxel_spacing) 
            
            assert np.sum(LETd_intpol_on_new_size) > 0, "No LETd values on CT!"
            
            print("Doses interpolated!")
            
            ct_intpol_on_new_size = change_voxel_spacing(array_CT, ct_voxel_spacing, new_voxel_spacing)
        
            # check spacings
            if np.shape(ct_intpol_on_new_size) != np.shape(dose_intpol_on_new_size) or \
                np.shape(ct_intpol_on_new_size) != np.shape(LETd_intpol_on_new_size) or \
                    np.shape(ct_intpol_on_new_size) != np.shape(check_ROI):
                bad_plans_save.append(plan)
                raise ValueError('CT, dose and ROI doesn\'t have the same shape!')
                
            print("Save files...")
            np.save(os.path.join(path_save_interpolated_data,plan, "plan_dose"),\
                    dose_intpol_on_new_size)
            np.save(os.path.join(path_save_interpolated_data,plan, "LETd"), \
                    LETd_intpol_on_new_size)
            np.save(os.path.join(path_save_interpolated_data,plan, "CT"), \
                    ct_intpol_on_new_size)
            print("Files saved!")
            
            # create figure to make plot for check
            fig, arr = plt.subplots()
            
            # plot slice, where MC dose has maximum
            max_dose_index = np.argmax(dose_intpol_on_new_size)
            max_dose_index = np.unravel_index(max_dose_index, dose_intpol_on_new_size.shape)
            plt.imshow(ct_intpol_on_new_size[max_dose_index[0],:,:], alpha = 1, cmap = "gray")
            plt.imshow(dose_intpol_on_new_size[max_dose_index[0],:,:], alpha = 0.8, cmap = "YlOrBr")
            plt.imshow(check_ROI[max_dose_index[0],:,:], alpha = 0.2, cmap = "gray")
            
            plt.savefig(os.path.join(path_save_check_figures, plan+".png"))
            
            print("Plot saved!")
            
            # save plan without error
            good_plans_save.append(plan)
        
        except Exception as e:
            print(e)
            bad_plans_save.append(plan)
            print("plan " + plan + " did not work!")
    
    update_bad_plans_list(bad_plans_save, good_plans_save, path_bad_plans)
    print("Following plans did not work: {}".format(bad_plans_save))
