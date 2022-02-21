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
cuda=8.0<br>
python=3.6<br>
torch=0.3.1<br>
torchvision=0.2.0<br>
cython<br>
opencv-python<br>
numpy<br>
scipy<br>
matplotlib<br>
pyyaml<br>
easydict<br>

Compilation
------
cd (root)/lib<br>
sh make.sh<br>

It will compile the NMS, ROI_Pooing, ROI_Crop and ROI_Align.<br>

Preparation
------
需要在根目录下新建一个data文件夹，并将以下数据集下载到对应位置。<br>
data：<br>
链接：https://pan.baidu.com/s/1Px4NsYUO41lPst1rfveQSA <br>
提取码：yv7t<br>

models：<br>
链接：https://pan.baidu.com/s/1LkG852BxkkTA9xTBknQ1yA <br>
提取码：ytw0<br>

normalset：<br>
链接：https://pan.baidu.com/s/1fxOeAvmS_QsVo2iIFDhZBA <br>
提取码：3732<br>

Training and Testing
------
本文用ResNet101作为预训练模型。如果你想重新训练模型并测试则需要在终端输入以下命令：<br>

第一个阶段的训练：<br>
CUDA_VISIBLE_DEVICES=0 python train_metarcnn.py --dataset pascal_voc_0712 --epochs 21 --bs 8 --nw 2 --log_dir checkpoint --save_dir models/meta/first --meta_train True --meta_loss True --lr_decay_step 10<br>

第二个阶段的训练：<br>
CUDA_VISIBLE_DEVICES=0 python train_metarcnn.py --dataset pascal_voc_0712 --epochs 71 --bs 4 --nw 1 --log_dir checkpoint --save_dir models/meta/first --r True --checksession 200 --checkepoch 20 --checkpoint 249 --phase 2 --shots 10 --meta_train True --meta_loss True --lr_decay_step 71<br>

测试：<br>
CUDA_VISIBLE_DEVICES=0 python test_metarcnn1.py --dataset pascal_voc_0712 --net TDENet --load_dir models/meta/first  --checksession 10 --checkepoch 51 --checkpoint 49 --shots 10 --meta_test True --meta_loss True --phase 2<br>

此外，我们也提供了第一个阶段训练好的模型 ，只需要将权重文件下载下来，放到对应文件夹下直接开始第二个阶段的训练即可。<br>


Contact
-------
If you have any questions, please contact me (chenzhao339@shu.edu.cn).<br>
