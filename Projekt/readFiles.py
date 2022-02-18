import SimpleITK as sitk, numpy as np, cv2 as cv

projections_file = 'projections.mhd'

itk_volume = sitk.ReadImage(projections_file)
volume = sitk.GetArrayFromImage(itk_volume)

for proj_idx in range(len(volume)):
    projection = volume[proj_idx]
    projection /= np.max(projection)

    cv.imshow('projection', projection)
    cv.waitKey()