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
The code of our TDENet has been made publicly available.

Requirements
------
cuda=8.0
python=3.6
torch=0.3.1
torchvision=0.2.0
cython
opencv-python
numpy
scipy
matplotlib
pyyaml
easydict

Compilation
------
cd (root)/lib
sh make.sh

It will compile the NMS, ROI_Pooing, ROI_Crop and ROI_Align.

Data Preparation
------
需要在根目录下新建一个data文件夹，并将以下数据集下载到对应位置。

Training and Testing
------
本文用ResNet101作为预训练模型。如果你想重新训练模型并测试则需要在终端输入以下命令：

第一个阶段的训练：


第二个阶段的训练：


测试：


此外，我们也提供了第一个阶段训练好的模型 ，只需要将权重文件下载下来，放到对应文件夹下直接开始第二个阶段的训练即可。

Contact
-------
If you have any questions, please contact me (chenzhao339@shu.edu.cn).
