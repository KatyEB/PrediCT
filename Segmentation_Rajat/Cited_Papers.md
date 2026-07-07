Agatston Scoring Rules & why is it important: https://www.jacc.org/doi/10.1016/j.jcmg.2022.02.026 (COCA_processor.py & agatston_script.py)

Co-ord Conv: https://papers.nips.cc/paper_files/paper/2018/file/60106888f8977b71e1f15db7bc9a88d1-Paper.pdf
The Paper Dicuss that Convolution networks who are Translation Invaraint, this inherent nature disables them to learn proper relationships between the pixel co-ordinates and output or input. Failing on tasks such a given a x,y coordinate generate a sqaure or given a sqaure give its co-ordinate. Paper gives a very smple idea of adding co-ordinate dimensions as input so that if the model wants to be translational invaraint it can be, by making the weights as zero but if it does not want then it can become trasnlation depedent. I beleive encoding information about the Locations will help us tell the model the prior locatinos of these lesions!. (dataset.py)

FNO for Parametric Partial Differntialbe Equations: https://arxiv.org/pdf/2010.08895  

nnUnet Paper: https://arxiv.org/pdf/1809.10486

For DOB-SCV: https://www.mdpi.com/1424-8220/23/4/2333, authour talk about how DOB_SCV which stands for Distribution-Balanced Stratified Cross-Validation is better thatn just SCV, in SCV which we all know we usally do this in you know classes where imbalance is high, to dont get under or over inflated results we distribute data for train aand val in a way that both have similar distributions in terms of target. we do not consider the features so there might be a chance that these two classes on which u r stratifiying thier clusters may overlap a lot that is 