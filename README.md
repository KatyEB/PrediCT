# PrediCT
A project to enhance predictive power of routine non-contrast CT scans

Problem:

Current coronary artery calcium (CAC) tests are widely used in clinical practice but have limited predictive power regarding when and where future occlusions may occur. While more advanced imaging techniques such as coronary CT angiography (CCTA) offer greater predictive capabilities and resolution, they are more resource-intensive, time-consuming, and less practical for routine use. At present, there is no diagnostic tool in cardiology that combines strong predictive ability with high clinical usability and accessibility.
________________________________________
Hypothesis:

Unsupervised machine learning models, when trained on a large set of CT scans, can detect predictive patterns in calcium deposition that are not readily identifiable by human interpretation. Such models could uncover hidden features that improve risk prediction beyond traditional scoring systems.
________________________________________
Approach:

1.	Dataset Collection

o	Obtain a large dataset of non-contrast CT scans through the Kettering Health Network.
o	Supplement with publicly available datasets if feasible.
o	Prioritize datasets with longitudinal scans where available.

3.	Preprocessing and Standardization

o	Standardize images for resolution, orientation, and noise reduction.
o	Seek expert guidance in medical image processing to ensure clinical relevance and consistency.

4.	Model Development
o	Construct an unsupervised learning model, with architecture informed by both clinical and technical input.
o	Begin with a basic framework, refining and adjusting based on iterative results.

6.	Training and Evaluation
o	Train and test the model on the available dataset, tuning parameters based on preliminary outcomes.
o	Explore the possibility of incorporating semi-supervised elements if appropriate.

7.	Integration with Biophysical Simulations (Long-Term)
o	Explore coupling model outputs with biophysical simulations of arterial stress and blood flow dynamics to enhance predictive insight.

8.	Clinical Translation (Long-Term Goal)

o	Develop a clinician-facing tool that, based on a simple CAC scan, outputs:
	Predicted time-dependent risk levels,
	Likely anatomical regions of future occlusion,
	Confidence intervals for predictions.
o	This tool could improve patient outcomes by assisting providers in balancing the risks of cardiac events with the risks of further tests, treatments, or surgeries.
o	Longer-term, this framework could serve as a model for applying machine learning to preventive medicine more broadly.
