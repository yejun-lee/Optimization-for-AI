import os 

os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

import time
from datetime import datetime, timedelta
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
torch.set_num_threads(4)
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torch_optimizer as optim

from sklearn.model_selection import train_test_split

import model

from tqdm import tqdm
from IPython.core.debugger import set_trace

if torch.cuda.is_available():
    print("Cuda is available !")
else:
    print("Running on CPU !")

print("\ndevice :", torch.cuda.get_device_name())
print("Number of detected GPU :", torch.cuda.device_count(), '\n')


def set_logger(save_path, log_name):
    os.makedirs(save_path, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] - %(message)s')

    file_handler = logging.FileHandler(os.path.join(save_path, log_name))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def plot_list(epoch, list, name, save_path):
    axis = np.linspace(1, epoch, epoch)
    
    plt.figure()
    plt.title(name)
    plt.plot(axis, list, label=name)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'{name}.png'))
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Optimization for AI Project')
    # train setting
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--optim', type=str, default="AdamP", choices=("A2GradExp", "A2GradInc", "A2GradUni", "AccSGD", "AdaBelief",
                                                        "AdaBound", "AdaMod", "Adafactor", "AdamP", "AggMo",
                                                        "Apollo", "DiffGrad", "LARS", "MADGRAD", "NovoGrad",
                                                        "PID", "QHAdam", "QHM", "RAdam", "Ranger",
                                                        "RangerQH", "RangerVA", "SGDP", "SGDW", "SWATS", "Yogi"))
    # model setting
    parser.add_argument('--model', type=str, default="EfficientNet", choices=("EfficientNet",  "VGG13",  "VGG16",  "VGG19",  "ResNet18", 
                                                                    "ResNet34",  "ResNet50",  "ResNet101",  "MobileNet_V2",  "MobileNet_V3_small", 
                                                                    "MobileNet_V3_large"))
    parser.add_argument('--in_ch', type=int, default=1)
    parser.add_argument('--out_ch', type=int, default=10)
    parser.add_argument('--pretrained', action='store_true')
    
    # log setting
    parser.add_argument('--save_path', type=str, default='weights/test')
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--log_name', type=str, default='train_log.log')
    
    # load setting
    parser.add_argument('--model_path', type=str, default=None)
    
    # seed
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    
    return parser.parse_args()


models = {"EfficientNet".lower():model.EfficientNet,
        "VGG13".lower():model.VGG13,
        "VGG16".lower():model.VGG16,
        "VGG19".lower():model.VGG19,
        "ResNet18".lower():model.ResNet18,
        "ResNet34".lower():model.ResNet34,
        "ResNet50".lower():model.ResNet50,
        "ResNet101".lower():model.ResNet101,
        "MobileNet_V2".lower():model.MobileNet_V2,
        "MobileNet_V3_small".lower():model.MobileNet_V3_small,
        "MobileNet_V3_large".lower():model.MobileNet_V3_large
    }


def main(args):
    # set_trace()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(f"{args.save_path}", exist_ok=True)
    
    fig_path = f"{args.save_path}/fig"
    ckpt_path = f"{args.save_path}/ckpt"
    os.makedirs(fig_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}")

    logger = set_logger(args.save_path, args.log_name)
    
    logger.info(args)
    print(args)

    dataset = torchvision.datasets.FashionMNIST(
                        root="data",
                        train=False,
                        download=True,
                        transform=torchvision.transforms.ToTensor()
                    )
    
    dataset_train, dataset_val = train_test_split(dataset, test_size=0.2, shuffle=False)
    print(f"Length of Train set: {len(dataset_train)}, Val set: {len(dataset_val)}")
    
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    loader_val = DataLoader(dataset_val, batch_size=200, shuffle=False, num_workers=8)

    net = models[args.model.lower()](args.in_ch, args.out_ch, args.pretrained)
    net.to(device)

    ##################################### Define Optimizer #####################################
    optims = {
        "A2GradExp".lower(): optim.A2GradExp,
        "A2GradInc".lower(): optim.A2GradInc,
        "A2GradUni".lower(): optim.A2GradUni,
        "AccSGD".lower(): optim.AccSGD,
        "AdaBelief".lower(): optim.AdaBelief,
        "AdaBound".lower(): optim.AdaBound,
        "AdaMod".lower(): optim.AdaMod,
        "Adafactor".lower(): optim.Adafactor,
        "AdamP".lower(): optim.AdamP,
        "AggMo".lower(): optim.AggMo,
        "Apollo".lower(): optim.Apollo,
        "DiffGrad".lower(): optim.DiffGrad,
        "LARS".lower(): optim.LARS,
        "MADGRAD".lower(): optim.MADGRAD,
        "NovoGrad".lower(): optim.NovoGrad,
        "PID".lower(): optim.PID,
        "QHAdam".lower(): optim.QHAdam,
        "QHM".lower(): optim.QHM,
        "RAdam".lower(): optim.RAdam,
        "Ranger".lower(): optim.Ranger,
        "RangerQH".lower(): optim.RangerQH,
        "RangerVA".lower(): optim.RangerVA,
        "SGDP".lower(): optim.SGDP,
        "SGDW".lower(): optim.SGDW,
        "SWATS".lower(): optim.SWATS,
        "Yogi".lower(): optim.Yogi,
    }

    if args.optim.lower() not in optims:
        raise ValueError(f"Unsupported optimizer: {args.optim}")
    
    optimizer = optims[args.optim.lower()](net.parameters(), lr=args.lr)
    ############################################################################################
    
    fn_loss = nn.CrossEntropyLoss().to(device)

    best_ce = 1e+6
    best_epoch = 0

    st_epoch = 1
    
    train_loss = []
    val_loss = []

    if args.model_path is not None:
        load_data = torch.load(args.model_path, map_location=device)
        st_epoch = load_data["epoch"] + 1

        net.load_state_dict(load_data["state_dict"])
        optimizer.load_state_dict(load_data["optimizer"])

        best_ce = load_data["best_ce"]
        best_epoch = load_data["best_epoch"]
        
        train_loss = load_data["train_log"].tolist()
        
        val_loss = load_data["val_log"].tolist()

        print("Model loaded !")
        print(f"Start epochs : {st_epoch}")


    for epoch in range(st_epoch, args.epochs+1):
        start = time.time()

        print(f"="*50)
        print(f"Time : {datetime.now()}")
        print()
        logger.info(f"="*50)
        logger.info(f"Time : {datetime.now()}\n")
        
        tmp_train_loss = []
        
        with tqdm(loader_train) as pbar:
            pbar.set_description(f"({epoch}/{args.epochs})")
            for data in loader_train:
                optimizer.zero_grad()
                
                ## Training SR Network
                net.train()
                
                inputs = data[0].to(device)
                labels = data[1].to(device)
                
                pred = net.forward(inputs)
                loss = fn_loss(pred, labels)
                loss.backward()
                
                tmp_train_loss += [loss.item()]
                
                optimizer.step()

                # pbar.set_postfix({"SR loss": np.round(np.mean(tmp_trainG_loss), 4)})
                pbar.set_postfix({"CE loss": np.round(np.mean(tmp_train_loss), 4)})
                pbar.update(1)
                
            train_epoch_loss = np.mean(tmp_train_loss)
            train_loss.append(train_epoch_loss)

            print(f"Train ({epoch}/{args.epochs}) Mean CE Loss : {train_epoch_loss:.4f}")
            logger.info(f"Train ({epoch}/{args.epochs}) Mean CE Loss : {train_epoch_loss:.4f}")
        
        # Validation
        with torch.no_grad():
            net.eval()
            
            tmp_val_ce = []

            with tqdm(loader_val) as pbar:
                for idx, data in enumerate(loader_val):
                    inputs = data[0].to(device)
                    labels = data[1].to(device)

                    pred = net.forward(inputs)
                    loss = fn_loss(pred, labels)
                    
                    tmp_val_ce += [loss.item()]

                    pbar.update(1)

            val_ce = np.mean(tmp_val_ce)
            val_loss.append(val_ce)

            print(f"Val ({epoch}/{args.epochs}) CE : {val_ce:.4f}")
            logger.info(f"Val ({epoch}/{args.epochs}) CE : {val_ce:.4f}")

            if val_ce < best_ce:
                best_epoch = epoch
                best_ce = val_ce
                
                torch.save({'epoch': epoch, 
                            'state_dict': net.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                            'best_score' : best_ce, 
                            'best_epoch' : best_epoch,
                            'train_log': torch.Tensor(train_loss),
                            'val_log': torch.Tensor(val_loss)
                            }, os.path.join(ckpt_path, f"model_best.pth"))

        if epoch % args.save_interval == 0:
            torch.save({'epoch': epoch, 
                        'state_dict': net.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'best_score' : best_ce, 
                        'best_epoch' : best_epoch,
                        'train_log': torch.Tensor(train_loss),
                        'val_log': torch.Tensor(val_loss)
                        }, os.path.join(ckpt_path, f"model_{epoch:04d}.pth"))
                    
        torch.save({'epoch': epoch, 
                    'state_dict': net.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'best_score' : best_ce, 
                    'best_epoch' : best_epoch,
                    'train_log': torch.Tensor(train_loss),
                    'val_log': torch.Tensor(val_loss)
                    }, os.path.join(ckpt_path, f"model_last.pth"))

        
        plot_list(epoch, train_loss, "train_loss", fig_path)
        plot_list(epoch, val_loss, "val_loss", fig_path)

        logger.info(f"\nTime spent on this epoch = {str(timedelta(seconds=time.time()-start)).split('.')[0]}")


if __name__=="__main__":
    args = parse_args()
    main(args)