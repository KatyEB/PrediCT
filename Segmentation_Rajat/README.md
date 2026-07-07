## Last Updated on: 17/06/2026 5:30PM IST

pre_process.py has been configured to generate ROI_Masks, which could be later cahced for faster infernce, These ROI masks will be used in Pipeline for ROI Masking and as a Input Channel, Obviously we will do ablation studies here so stay tuned...

dataset.py has been configured to include Co-ord convolution channels, Heart ROI masks, DUAL HU Windowing for both Calcium and Tissues. Persistent Caching to speed up training



  