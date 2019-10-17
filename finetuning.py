from tqdm import tqdm
import torch
import torch.nn as nn
from utils import accuracy

def train_batch(model, criterion, optimizer, train_loader, scheduler=None, clip=0.25, device=None, lr_decrease=2):
    train_loss = 0
    train_acc = 0
    
    model.train()
    model.reset_hidden()
    with tqdm(total=len(train_loader)) as t:
        for batch in train_loader:
            x, y = batch
            inputs = x.permute(1, 0).to(device)
            targets = y.to(device)

            # Adjust discriminative learning rates
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] /= lr_decrease ** i
                    
            out = model(inputs)
            loss = criterion(out, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            t.set_postfix({'lr{}'.format(i): optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))})
            scheduler.step()

            train_loss += loss.item()
            train_acc += accuracy(out, targets)
            
            t.update()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    
    return train_loss, train_acc

def eval_batch(model, criterion, val_loader, device=None):
    val_loss = 0
    val_acc = 0
    
    model.eval()
    model.reset_hidden()
    for batch in tqdm(val_loader):
        with torch.no_grad():
            x, y = batch
            inputs = x.permute(1, 0).to(device)
            targets = y.to(device)

            out = model(inputs)
            loss = criterion(out, targets)

            val_loss += loss.item()
            val_acc += accuracy(out, targets)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    
    return val_loss, val_acc

def one_cycle(model, criterion, optimizer, train_loader, val_loader, scheduler=None, clip=0.25, device=None, lr_decrease=2):
    train_loss, train_acc = train_batch(model, criterion, optimizer, train_loader, scheduler=scheduler, clip=clip, device=device, lr_decrease=lr_decrease)
    val_loss, val_acc = eval_batch(model, criterion, val_loader, device=device)
    print("Train Loss: {:.4f} | Train Acc: {:.4f} | Val Loss: {:.4f} | Val Acc: {:.4f}".format(train_loss, train_acc, val_loss, val_acc))