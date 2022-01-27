from torchvision import models
import torch
import torchattacks
import torch.nn as nn
import dataset
from collections import Counter
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from cifar10_models.resnet import ResNet18,ResNet18_reg, ResNet34, ResNet50,PreActResNet18,PreActResNet18_reg
import argparse
from tqdm import tqdm
import torchvision
import pdb
import hook
import easypyxl
from torch.autograd import Variable
import torch.nn.functional as F
from cifar10_models.wideresnet import WideResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _cw_whitebox(model,
                  X,
                  y,
                  epsilon=8/255,
                  num_steps=20,
                  step_size=0.003,
                  beta=2.0):
    # out = model(X)
    # err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            output = model(X_pgd )
            correct_logit = torch.sum(torch.gather(output, 1, (y.unsqueeze(1)).long()).squeeze())
            tmp1 = torch.argsort(output, dim=1)[:, -2:]
            new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
            wrong_logit = torch.sum(torch.gather(output, 1, (new_y.unsqueeze(1)).long()).squeeze())
            loss = - F.relu(correct_logit - wrong_logit)

        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    return X_pgd



class Normalization(nn.Module):
    def __init__(self, mean, std,device):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.mean = self.mean.to(device, torch.float)
        self.std = self.std.to(device, torch.float)

    def forward(self, img):
        return (img - self.mean) / self.std



def main(args): 
    best_adv_acc, best_clean_acc = 0,0
    best_fgsm_acc = best_pgd_acc = best_cw_acc = 0.0
    train_set,test_set = dataset.CIFAR10(normalize=False,download=False)
    num_classes = 10
    batch_size = int(args.batch_size)
    epoches = 100
    train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size = batch_size,shuffle = False)
    layer_idx = int(args.layer_idx)
    alpha = float(args.alpha)
    training_method = args.training_method
    reg_loss = nn.MSELoss() if args.reg_loss == 'l2_loss' else nn.L1Loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = vgg13_bn(pretrained=True if args.pre_trained else False)
    
    if args.model_name == 'resnet18':
        model = ResNet18()
    elif args.model_name == 'resnet18_reg':
        model = ResNet18_reg()
    elif args.model_name == 'preAct_resnet18':
        model = PreActResNet18()
    elif args.model_name == 'preAct_resnet18_reg':  
        model = PreActResNet18_reg()      
    elif args.model_name == 'WRN':
        model = WideResNet()
    elif args.model_name == 'WRN_reg':
        model = WideResNet(reg = True)


    f = open("./results/" + args.model_name + " " + args.training_method + " "  + str(batch_size) + ".txt",'a')
    f.write("model : %s, training_method : %s, loss : %s, alpha : %.3f, epoch : %d, batch size: %d num_classes : %d \n"%(args.model_name,args.training_method, args.reg_loss, alpha, epoches, batch_size,num_classes))
    print ("model : %s, training_method : %s, loss : %s, alpha : %.3f, epoch : %d, batch size: %d num_classes : %d"%(args.model_name,args.training_method, args.reg_loss, alpha, epoches, batch_size,num_classes))

    model = nn.Sequential(
        Normalization([0.485,0.456,0.406],[0.229,0.224,0.225],device),
        model
        )

    model = model.to(device)
    if args.training_method == 'AT' or args.training_method == 'MART':
        train_attack = torchattacks.PGD(model, eps = 8/255,steps = 7)
    elif args.training_method == 'TRADES':
        train_attack = torchattacks.TPGD(model,eps = 8/255)

    test_attack = torchattacks.PGD(model,eps = 8/255,steps = 20)
    lr = 1e-2
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay=0.0002)
    clean_acc = adv_acc = 0
    for epoch in range(epoches):
        adjust_learning_rate(lr, optimizer, epoch)
        train(model,train_loader,optimizer,train_attack,alpha,reg_loss,args.training_method)
        clean_acc, fgsm_acc, pgd_acc, cw_acc = evaluate(model,test_loader,test_attack,epoch,f)
        # adv_acc = (fgsm_acc + pgd_acc + cw_acc)/3.
        adv_acc = pgd_acc
        if best_adv_acc < adv_acc:
            best_clean_acc = clean_acc
            best_adv_acc = adv_acc
            best_fgsm_acc = fgsm_acc
            best_pgd_acc = pgd_acc
            best_cw_acc = cw_acc
            torch.save(model.state_dict(),args.best_model_path)
        

    print (f"Best_clean_acc : {best_clean_acc:>.3f},Best_robust_acc : {best_adv_acc:>.3f}, Best_fgsm_acc : {best_fgsm_acc:>.3f}, Best_pgd_acc : {best_pgd_acc:>.3f}, Best_cw_acc : {best_cw_acc:>.3f}")
    print (f"Last_clean_acc : {clean_acc:>.3f}, Last_robust_acc : {adv_acc:>.3f}, Last_fgsm_acc : {fgsm_acc:>.3f}, Last_pgd_acc : {pgd_acc:>.3f}, Last_cw_acc : {cw_acc:>.3f} ")
    print ("="* 20)
    f.write(f"Best_clean_acc : {best_clean_acc:>.3f},Best_robust_acc : {best_adv_acc:>.3f}, Best_fgsm_acc : {best_fgsm_acc:>.3f}, Best_pgd_acc : {best_pgd_acc:>.3f}, Best_cw_acc : {best_cw_acc:>.3f} \n")
    f.write(f"Last_clean_acc : {clean_acc:>.3f}, Last_robust_acc : {adv_acc:>.3f}, Last_fgsm_acc : {fgsm_acc:>.3f}, Last_pgd_acc : {pgd_acc:>.3f}, Last_cw_acc : {cw_acc:>.3f} \n ")

    torch.save(model.state_dict(),args.last_model_path)


def PGD_attack_untargeted(model, x,label_y,eps = 8/255, steps = 1):
    delta =  2*eps*torch.rand_like(x) - eps
    alpha = eps/steps
    for k in range(steps):
        delta.requires_grad_()
        preds = model(x + delta)
        loss = nn.CrossEntropyLoss()(preds,label_y)
        grad = torch.autograd.grad(loss, [delta])[0]
        delta = delta.detach() + alpha * torch.sign(grad.detach())
        delta = torch.clip(delta,-eps,eps)
    
    x_adv = torch.clip(x+delta,0,1)
    return x_adv



def train(model,train_loader,optimizer,train_attack,alpha,reg_loss,training_method = None):
    '''
    model : torch model
    layer_idx : int or list 
    '''
    model.train()
    idx = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    # result = LayerResult(model[1].features,layer_idx)
    clean_acc = 0
    adv_acc = 0
    total_samples = 0
    train_loss = 0
    correct = 0
    features = hook.FeatureExtractor(model,0)
    
    if training_method == 'TRADES' :
        beta = 6.0
        criterion_kl = nn.KLDivLoss(size_average=False)
    elif training_method == 'MART' :
        beta = 5.0
        criterion_kl = nn.KLDivLoss(reduction='none')

    for batch_idx, (x,y) in tqdm(enumerate(train_loader)):
        batch_size = x.shape[0]
        total_samples += batch_size
        optimizer.zero_grad()
        x,y = x.to(device),y.to(device)
        # z_clean = model(x)
        z_clean = features(x)
        activations_clean = features.activations.copy()

        N,C,H,W = x.shape
        x_adv = train_attack(x,y)

        # x_adv = PGD_attack_untargeted(model,x,y)
        z_adv = features(x_adv)
        activations_adv = features.activations.copy()

        ce_adv_loss = criterion(z_adv,y)
        feature_loss = 0

        if alpha > 0 :
            for (clean_feature,adv_feature) in zip(activations_clean.values(),activations_adv.values()):
                feature_loss += reg_loss(clean_feature.mean(dim = (2,3)),adv_feature.mean(dim  =(2,3)))

        M = len(activations_clean)
        alpha = alpha *(1/M)

        if training_method == 'AT':
            loss = ce_adv_loss + alpha *feature_loss

        elif training_method == 'TRADES':
            z = model(x)
            loss_natural = criterion(z, y)
            loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(x), dim=1))
            loss = loss_natural + beta * loss_robust + alpha * feature_loss

        elif training_method == 'MART':
            
            z = model(x)
            adv_probs = F.softmax(z_adv, dim=1)
            tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

            new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
            loss_adv = F.cross_entropy(z_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
            nat_probs = F.softmax(z, dim=1)
            true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

            loss_robust = (1.0 / batch_size) * torch.sum(torch.sum(criterion_kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
            loss = loss_adv + float(beta) * loss_robust + alpha * feature_loss
    

        loss.backward()
        optimizer.step()

        z_clean_out = z_clean.argmax(dim = 1)
        z_adv_out = z_adv.argmax(dim = 1)
        clean_acc = (z_clean_out ==y).sum()
        adv_acc = (z_adv_out == y).sum()

        # if batch_idx % 10 == 0 : 
        #     print (f"clean_acc: {clean_acc/total_samples :>.3f}, adv_acc : {adv_acc/total_samples :>.3f}")


def evaluate(model,test_loader,attack,epoch,f):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_acc = 0
    fgsm_acc = pgd_acc = cw_acc = 0
    total_samples = 0
    fgsm_attack = torchattacks.FGSM(model,eps = 8/255)
    pgd_attack = torchattacks.PGD(model,eps = 8/255,steps = 20)
    # cw_attack = torchattacks.CW(model, c=1, kappa=0, steps=20, lr=0.01)
    # aa_attack = torchattacks.AutoAttack(model,eps = 8/255)


    for i,(x,y) in enumerate(test_loader):
        total_samples += x.shape[0]
        x,y = x.to(device), y.to(device)
        
        x_fgsm = fgsm_attack(x,y)
        x_pgd = pgd_attack(x,y)
        x_cw = _cw_whitebox(model,x,y)
        # x_cw = pgd_attack(x,y)

        

        z_clean = model(x)
        z_fgsm = model(x_fgsm)
        z_pgd = model(x_pgd)
        z_cw = model(x_cw)

        z_clean_out = z_clean.argmax(dim = 1)
        z_fgsm_out = z_fgsm.argmax(dim = 1)
        z_pgd_out = z_pgd.argmax(dim = 1)
        z_cw_out = z_cw.argmax(dim = 1)

        clean_acc += (z_clean_out ==y).sum()
        fgsm_acc += (z_fgsm_out == y).sum()
        pgd_acc += (z_pgd_out == y).sum()
        cw_acc += (z_cw_out == y).sum()
        # print (f"epoch : {epoch : d}, clean_acc: {clean_acc/total_samples :>.3f}, adv_acc : {adv_acc/total_samples :>.3f}")
        # print (i,)
     
    clean_acc = clean_acc/total_samples
    fgsm_acc = fgsm_acc/total_samples
    pgd_acc = pgd_acc/total_samples
    cw_acc = cw_acc/total_samples
    # adv_acc = adv_acc/total_samples
    if (epoch+1) % 10 == 0 :
        print (f"epoch : {epoch + 1: d}, clean_acc: {clean_acc :>.3f}, fgsm_acc : {fgsm_acc :>.3f}, pgd_acc : {pgd_acc :>.3f}, cw_acc : {cw_acc :>.3f}",flush = True)
        f.write(f"epoch : {epoch + 1: d}, clean_acc: {clean_acc :>.3f}, fgsm_acc : {fgsm_acc :>.3f}, pgd_acc : {pgd_acc :>.3f}, cw_acc : {cw_acc :>.3f},\n")
   
    return clean_acc, fgsm_acc, pgd_acc, cw_acc

def adjust_learning_rate(lr, optimizer, epoch):
    if epoch >= 75:
        lr /= 10
    if epoch >= 90:
        lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def argument_parsing():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", default="resnet18")
    parser.add_argument("--batch_size", default=128, help="batch_size")
    parser.add_argument("--layer_idx", default=2, help="layer_index")
    parser.add_argument("--alpha", default=0, help="layer_index")
    parser.add_argument("--training_method", default = "AT", help = "TRADES or MART")
    parser.add_argument("--reg_loss", default = 'l1_loss',help="l2_loss or l1_loss")
    parser.add_argument("--use_feature", dest = 'use_feature', action = 'store_true')
    parser.add_argument("--pre_trained", dest = 'pre_trained', action = 'store_true')
    parser.add_argument("--file_idx", default = '1')
    parser.add_argument("--best_model_path", default = './check_point/best_model.pth')
    parser.add_argument("--last_model_path", default = './check_point/last_model.pth')

    return parser

if __name__ == "__main__":
    args = argument_parsing().parse_args()
    main(args)
