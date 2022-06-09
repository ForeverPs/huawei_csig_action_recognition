import os
import tqdm
import torch
import shutil
import torch.nn as nn
from torch import optim
from utils import get_acc
from data import data_pipeline
from model.model import ActNet
from tensorboardX import SummaryWriter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(epochs, lr, length):
    data_loader = data_pipeline(txt_path, data_prefix, aug_ratio=0.5)
    val_loader = data_pipeline(txt_path, data_prefix, aug_ratio=0)
    model = ActNet(ActionLength=length)

    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    except:
        print('Training from scratch...')

    optimizer = optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_loss = 0.8, 0.5
    for epoch in range(epochs):
        train_loss, val_loss = 0, 0
        train_acc, val_acc = 0, 0
        
        model.train()
        for x, y in tqdm.tqdm(data_loader):
            x = x.squeeze(0).float().to(device) 
            y = y.long().to(device)  

            predict = model(x)
            loss = criterion(predict, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()
            _, predict_cls = torch.max(predict, dim=-1)
            train_acc += get_acc(predict_cls, y)
        
        # update learning rate
        scheduler.step()
            
        model.eval()
        with torch.no_grad():
            for x, y in tqdm.tqdm(val_loader):
                x = x.squeeze(0).float().to(device)
                y = y.long().to(device) 
                predict = model(x)
                loss = criterion(predict, y)
                val_loss = val_loss + loss.item()
                _, predict_cls = torch.max(predict, dim=-1)
                val_acc += get_acc(predict_cls, y)
        
        train_loss = train_loss / len(data_loader)
        train_acc = train_acc / len(data_loader)

        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)
        
        print('EPOCH : %03d | Train Loss : %.3f | Train Acc : %.3f | Val Loss : %.3f | Val Acc : %.3f' % 
              (epoch, train_loss, train_acc, val_loss, val_acc))
        
        compare_acc = (val_acc + train_acc) / 2.0
        compare_loss = (val_loss + train_loss) / 2.0
        if compare_acc >= best_acc and compare_loss <= best_loss:
            best_acc = compare_acc
            best_loss = compare_loss
            model_name = 'epoch_%d_acc_%.3f.pth' % (epoch, best_acc)
            os.makedirs(saved_path, exist_ok=True)
            torch.save(model.state_dict(), '%s/%s' % (saved_path, model_name))
        
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)

        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', val_acc, epoch)


if __name__ == '__main__':
    length = 800
    logdir='./tensorboard/'
    saved_path = './saved_models/800_1280/'
    shutil.rmtree(logdir, True)
    writer = SummaryWriter(logdir)

    lr = 2e-5
    txt_path = './data/label.txt'
    data_prefix = './data/npy_data_6d/'
    # model_path = './saved_models/800_2048/epoch_50_acc_1.000.pth'
    model_path = './saved_models/800_1280/epoch_57_acc_1.000.pth'
    train(epochs=100, lr=lr, length=length)
