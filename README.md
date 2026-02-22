# PrediCT
A project to enhance predictive power of routine non-contrast CT scans

Problem:

Current coronary artery calcium (CAC) tests are widely used in clinical practice but have limited predictive power regarding when and where future occlusions may occur. While more advanced imaging techniques such as Intravascular Ultrasound (IVUS) and optical coherence tomography (OCT) offer greater predictive capabilities and resolution, they are more resource-intensive, time-consuming, and less practical for routine use. At present, there is no diagnostic tool in cardiology that combines strong predictive ability with high clinical usability and accessibility.
________________________________________
Hypothesis:

Machine learning models, when trained on a large set of CT scans, can detect predictive patterns in calcium deposition that are not readily identifiable by human interpretation. Such models could uncover hidden features that improve risk prediction beyond traditional scoring systems, discover correlations between clinical and image data, and enhance the predictive capability of these scans through powerful feature detection.
________________________________________
Current Direction:

While we await data transfer of major adverse cardiovascular event (MACE) endpoints + NCCT from Kettering Health Network, we will focus on constructing a robust front end for our model using open source data (Stanford COCA, ImageCAS, etc.). This includes a calcium segmentation head that combines speed with accuracy while preserving accurate spatial relations. We also aim to map the heart anatomy to properly localize calcium deposits. We will perform extensive feature extraction and analysis, with the goal of creating distinct calcium phenotypes that may map to MACE endpoints. Eventually, with expert annotation, we hope to integrate other features such as EAT, Heart Chamber Volume, etc. To supplement our current efforts we are also exploring data augmentation using simulation-based synthetic calcium generation and placement into empty CAC scans.

Clinical Translation (Long-Term Goal)

o	Develop a clinician-facing tool that, based on a simple CAC scan, outputs:

	Predicted time-dependent risk levels for MACE,
	Likely anatomical regions of future occlusion?,
	Confidence intervals for predictions.

o	This tool could improve patient outcomes by assisting providers in balancing the risks of cardiac events with the risks of further tests, treatments, or surgeries.

o	Longer-term, this framework could serve as a model for applying machine learning to preventive medicine more broadly.
