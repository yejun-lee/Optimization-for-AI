This repository is for the team project of the Optimization for AI course.

Team members: Yejun Lee, Sanghoon Shin, Chan Jang, Seojun Kim

**Here are the commands to install required modules**
```markdown
conda create -n optim python=3.8
conda activate optim
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch_optimizer
git clone https://github.com/yejun-lee/Optimization-for-AI
```

**In train.sh, you can see the available optimizers and models as well as the examples of training code.**
```shell
############################ Avaliable optims ################################
#    "A2GradExp" "A2GradInc" "A2GradUni" "AccSGD" "AdaBelief" "AdaBound"     #
#    "AdaMod" "Adafactor" "AdamP" "AggMo" "Apollo" "DiffGrad"                #
#    "LARS" "Lamb" "MADGRAD" "NovoGrad" "PID" "QHAdam"                       #
#    "QHM" "RAdam" "Ranger" "RangerQH" "RangerVA" "SGDP"                     #
#    "SGDW" "SWATS" "Shampoo" "Yogi" "Lion"                                  #
##############################################################################

################################ Avaliable models ###################################
#  "EfficientNet" "VGG13" "VGG16" "VGG19" "ResNet18" "ResNet34"                     #
#  "ResNet50" "ResNet101" "MobileNet_V2" "MobileNet_V3_small" "MobileNet_V3_large"  #
#####################################################################################


python train.py --gpu 6 --model EfficientNet --save_path weights/A2GradExp --optim A2GradExp --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/A2GradInc --optim A2GradInc --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/A2GradUni --optim A2GradUni --batch_size 256 --lr 4e-2 --epochs 300

...
```

Set your desired model, optimizer, and training parameters, and run the code.
