File Struct:

COCA_Setup_Files\
	COCA_pipeline.py
	COCA_processor.py
	COCA_resampler.py
	README.md
	Documentation.md
	Utilities\
		COCA_Sample_Dicom_Metadata.md
		COCA_Voxel_Spacing.md
		UnLabelled_Arteries.md
		get_image_sizes.py
		visualizer_3D.py


DETAILED INFORMATION: 

COCA_pipeline.py : Runs both Processor and Resampler

COCA_processor.py : Process and Combines all the .dcm files of patients and also generates their respective artery masks and stores them in data_canonical folder

COCA_resampler.py : Resample to a uniform Voxel Spacing

README.md : All the Instructions to setup the Dataset and make it accessible

Documentation.md : Documenting relevant information for Development, Usage, Debugging, Logging etc

Utitlies\COCA_Sample_Dicom_Metadata.md : Shows original meta data of dicom file

Utilities\UnLabelled_Arteries.md : File Locations which contain unlabelled arterires 

Utilities\COCA_Voxel_Spacing.txt : Has information about Voxel spacings of each patinet's CT Scans

Utilities\Visualizer_3D.py : Visualize 3D Files using GPU 

Utilities\get_image_sizes.py : Get image sizes of all volumes

INSTRUCTIONS ON HOW TO USE:

Before this you must have the COCA_Dataset downloaded from https://stanfordaimi.azurewebsites.net/datasets/e8ca74dc-8dd4-4340-815a-60b41f6cb2aa 

Paste the 3 Mail files resampler, pipeline, processor in the folder which contains the dataset folder as show in "setup_image.png" and then run the COCA_pipeline.py where you have to first paste the location where you will store the folder which contains the proceed images and binary + multi masks. Just run the file, its super user friendly

WORK LEFT TO DO:
1. Inspect why id no 263 is not able to get its binary segmentation!
2. Interpolation is not invertible is my guess, because to maintain 100% accruacy in scores, scores before and after resampling must be same, so i have to look into it as well

UPDATES FROM PREV SCRIPTS:

1. Modified the Scritps to Calculate the Agtatson scores from the Area and HU information from the XML Files
2. Now we get Multi Segmented Labels, If we had to do Multi Segmentation in Future scope of PrediCT
3. The Processors actually dint take into account of mutilple series in dicom files, patinet id 135 and 763 had 2 series in them one of which had only 1 z slice, but the 2nd series had more Z slcies which looked more realistic so i edited the code to catch all possible series and keep the one which has more z slices
5. Utilities Folder contains Scripts that help in exploring and doing reserach, and also details about the COCA dataset that may help
4. Few edits to datapipeline and small intructions on how to use the file


