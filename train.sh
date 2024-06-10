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
# python train.py --gpu 6 --model EfficientNet --save_path weights/AccSGD --optim AccSGD --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/AdaBelief --optim AdaBelief --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/AdaBound --optim AdaBound --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/AdaMod --optim AdaMod --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/Adafactor --optim Adafactor --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/AdamP --optim AdamP --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/AggMo --optim AggMo --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/Apollo --optim Apollo --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/DiffGrad --optim DiffGrad --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/LARS --optim LARS --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/Lamb --optim Lamb --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/MADGRAD --optim MADGRAD --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/NovoGrad --optim NovoGrad --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/PID --optim PID --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/QHAdam --optim QHAdam --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/QHM --optim QHM --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/RAdam --optim RAdam --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/Ranger --optim Ranger --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/RangerQH --optim RangerQH --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/RangerVA --optim RangerVA --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/SGDP --optim SGDP --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/SGDW --optim SGDW --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/SWATS --optim SWATS --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/Shampoo --optim Shampoo --batch_size 256 --lr 4e-2 --epochs 300
# python train.py --gpu 6 --model EfficientNet --save_path weights/Yogi --optim Yogi --batch_size 256 --lr 4e-2 --epochs 300
