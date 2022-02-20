# Two-phase defect prominence network for few-shot defect detection with defect-free samples
This repository contains a new ISD dataset and a novel TDENet for Few-Shot Defect Detection.

Original datasets
------
The NEU dataset can be found in http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html. <br>
The MT dataset can be found in  https://github.com/abin24/Saliency-detection-toolbox. <br>

ISD dataset
------
There is no dataset for Few-Shot Defect Detection (FSDD) at present. We construct a new dataset (ISD dataset) to boost the development of FSDD. Our new dataset is derived from two defect datasets, i.e., the NEU dataset and the MT dataset. The NEU dataset is for steel, which is suitable for the study of defect detection.However, the MT dataset is made for the image saliency detection. We transform the binary ground truths to the defect detection annotations as follows:<br>
(1)Individual defect acquisition.<br>
(2)Defect detection annotations generation. <br>
In this way, we complete the transformation of the MT dataset.

Code for TDENet
------
The code of our TDENet will be made publicly available upon the acceptance of the paper.

Contact
-------
If you have any questions, please contact me (chenzhao339@shu.edu.cn).
