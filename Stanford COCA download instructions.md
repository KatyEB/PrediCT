Hello interested contributors! Here’s how to download the dataset from COCA. Unfortunately, we cannot directly deliver data because of Stanford’s Terms of Use. Each person must individually upload and parse the dataset. However, we will provide the tools with which to do so!

1.	Go to Stanford AIMI datasets, find COCA, or follow this link: 

https://stanfordaimi.azurewebsites.net/

2.	Make an account with AIMI, login to view the dataset, accept the license

3.	Download the dataset using one of the available options

4.	Download the following scripts to the same folder. They are in the COCA_scripts zip file in this github: KatyEB/PrediCT: A project to enhance predictive power of routine non-contrast CT scans

  a.	Unnester: Fixes the format of the gated files to remove the intermediate scan label folder 
  
  b.	COCA_processor: class that creates 3D files of both the DICOM image and segmentations, as well as an output table

  c.	COCA_resampler: class that resamples the voxels to what you would like (recommended .7 x .7 x 3 mm)

  d.	COCA_pipeline: runs COCA processor and resampler

6.	Run Unnester to fix the Gated dataset

7.	Make sure COCA_processor and COCA_resampler are set up correctly to run in COCA_pipeline, taking care to tailor them to your file organization
  
8.	Run COCA_pipeline

9.	If you would like to view the files, I recommend downloading 3D Slicer, and uploading the nii image and segmentation zip files together using the DATA upload option. Make sure to indicate the segmentation file is a segmentation rather than a volume.
