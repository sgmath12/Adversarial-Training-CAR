import torch
import torchattacks
import torch.nn as nn
import dataset
import torch.optim as optim
from cifar10_models.resnet import ResNet18,ResNet18_reg
import argparse
from tqdm import tqdm
import hook
from utils import _cw_whitebox, Normalization
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args): 

    num_classes = int(args.num_classes)
    if num_classes == 10 and args.dataset == 'cifar':
        train_set,test_set = dataset.CIFAR10(root = './dataset',normalize=False,download=False)
    elif num_classes == 100:
        train_set,test_set = dataset.CIFAR100(root = './dataset',normalize=False,download=False)
        
    batch_size = int(args.batch_size)
    epoches = int(args.epoches)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size = batch_size,shuffle = False)
    alpha = float(args.alpha)
    lr = float(args.lr)
    training_method = args.training_method
    reg_loss = nn.MSELoss() if args.reg_loss == 'l2_loss' else nn.L1Loss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_name == 'resnet18':
        model = ResNet18(num_classes = num_classes)
    elif args.model_name == 'resnet18_reg':
        model = ResNet18_reg(num_classes = num_classes)


    print ("model : %s, training_method : %s, loss : %s, alpha : %.3f, epoches : %d, batch size: %d , num_classes : %d , lr : %.2f , load : %s"%(args.model_name,args.training_method, args.reg_loss, alpha, epoches, batch_size,num_classes,lr, args.load), flush = True)
    mean = torch.tensor([0.485,0.456,0.406], device = device)
    std = torch.tensor([0.229,0.224,0.225], device = device)

    model = nn.Sequential(
        Normalization(mean,std),
        model
        )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if args.training_method == 'AT' or args.training_method == 'MART':
        train_attack = torchattacks.PGD(model, eps = 8/255,steps = 7)
        beta = 6.0
        criterion_kl = nn.KLDivLoss(reduction='none')
    elif args.training_method == 'TRADES':
        train_attack = torchattacks.TPGD(model,eps = 8/255,steps = 7)
        beta = 5.0
        criterion_kl = nn.KLDivLoss(size_average=False)


    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay=0.0002)
    for epoch in range(epoches):
        adjust_learning_rate(lr, optimizer, epoch)
        train(model,train_loader,optimizer,train_attack,alpha,beta,reg_loss,criterion,criterion_kl,args.training_method)

    torch.save(model.state_dict(),args.last_model_path)
    last_clean_acc, last_fgsm_acc, last_pgd_acc, last_cw_acc, last_aa_acc  = evaluate(model,test_loader)
    print (f"Last_clean_acc : {last_clean_acc:>.5f}, Last_fgsm_acc : {last_fgsm_acc:>.5f}, Last_pgd_acc : {last_pgd_acc:>.5f}, Last_cw_acc : {last_cw_acc:>.5f}, Last_aa_acc : {last_aa_acc:>.5f}  ",flush = True)


def train(model,train_loader,optimizer,train_attack,alpha,beta,reg_loss,criterion,criterion_kl,training_method = None):
    '''
    model : torch model
    layer_idx : int or list 
    '''
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    total_samples = 0
    features = hook.FeatureExtractor(model,0)
    

    for batch_idx, (x,y) in enumerate(train_loader):
        batch_size = x.shape[0]
        total_samples += batch_size
        optimizer.zero_grad()
        x,y = x.to(device),y.to(device)

        z_clean = features(x)
        activations_clean = features.activations.copy()

        N,C,H,W = x.shape
        x_adv = train_attack(x,y)

        z_adv = features(x_adv)
        activations_adv = features.activations.copy()

        ce_adv_loss = criterion(z_adv,y)
        feature_loss = 0

        if alpha > 0 :
            for (clean_feature,adv_feature) in zip(activations_clean.values(),activations_adv.values()):
                feature_loss += reg_loss(clean_feature.mean(dim = (2,3)),adv_feature.mean(dim  =(2,3)))

        M = len(activations_clean) 
        if training_method == 'AT':
            loss = ce_adv_loss + alpha * (1/M) * feature_loss

        elif training_method == 'TRADES':
            z = model(x)
            loss_natural = criterion(z, y)
            loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1), F.softmax(model(x), dim=1))
            loss = loss_natural + beta * loss_robust + alpha * (1/M) * feature_loss

        elif training_method == 'MART':
            z = model(x)
            adv_probs = F.softmax(z_adv, dim=1)
            tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

            new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
            loss_adv = F.cross_entropy(z_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
            nat_probs = F.softmax(z, dim=1)
            true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

            loss_robust = (1.0 / batch_size) * torch.sum(torch.sum(criterion_kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
            loss = loss_adv + float(beta) * loss_robust + alpha * (1/M) * feature_loss
    

        loss.backward()
        optimizer.step()

        # if batch_idx % 10 == 0 : 
        #     print (f"clean_acc: {clean_acc/total_samples :>.3f}, adv_acc : {adv_acc/total_samples :>.3f}")

def evaluate(best_model,test_loader):
    best_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_acc = 0
    fgsm_acc = cw_acc = aa_acc = pgd_acc =  0
    total_samples = 0
    fgsm_attack = torchattacks.FGSM(best_model,eps = 8/255)
    pgd_attack = torchattacks.PGD(best_model,eps = 8/255,steps = 20)
    aa_attack = torchattacks.AutoAttack(best_model, eps = 8/255)

    for i,(x,y) in enumerate(test_loader):
        total_samples += x.shape[0]
        x,y = x.to(device), y.to(device)
        
        x_fgsm = fgsm_attack(x,y)
        x_pgd = pgd_attack(x,y)
        x_cw = _cw_whitebox(best_model,x,y)
        x_aa = aa_attack(x,y)


        z_clean = best_model(x)
        z_fgsm = best_model(x_fgsm)
        z_pgd = best_model(x_pgd)
        z_cw = best_model(x_cw)
        z_aa = best_model(x_aa)


        z_clean_out = z_clean.argmax(dim = 1)
        z_fgsm_out = z_fgsm.argmax(dim = 1)
        z_pgd_out = z_pgd.argmax(dim = 1)
        z_cw_out = z_cw.argmax(dim = 1)
        z_aa_out = z_aa.argmax(dim = 1)


        clean_acc += (z_clean_out ==y).sum()
        fgsm_acc += (z_fgsm_out == y).sum()
        pgd_acc += (z_pgd_out == y).sum()
        cw_acc += (z_cw_out == y).sum()
        aa_acc += (z_aa_out == y).sum()

     
    clean_acc = clean_acc/total_samples
    fgsm_acc = fgsm_acc/total_samples
    pgd_acc = pgd_acc/total_samples
    cw_acc = cw_acc/total_samples
    aa_acc = aa_acc/total_samples
   
    return clean_acc, fgsm_acc, pgd_acc, cw_acc, aa_acc

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
    parser.add_argument("--num_classes", default=10, help="num classes")
    parser.add_argument("--epoches", default=100, help="batch_size")
    parser.add_argument("--dataset", default='cifar', help="layer_index")
    parser.add_argument("--alpha", default=0, help="layer_index")
    parser.add_argument("--training_method", default = "AT", help = "TRADES or MART")
    parser.add_argument("--reg_loss", default = 'l1_loss',help="l2_loss or l1_loss")
    parser.add_argument("--lr", default = '0.01')
    parser.add_argument("--best_model_path", default = './check_point/best_model.pth')
    parser.add_argument("--last_model_path", default = './check_point/last_model.pth')
    parser.add_argument("--load", default='False')


    return parser

if __name__ == "__main__":
    args = argument_parsing().parse_args()
    main(args)
