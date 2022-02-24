from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def _cw_whitebox(model,
                  X,
                  y,
                  epsilon=8/255,
                  num_steps=20,
                  step_size=0.003,
                  beta=2.0):
    # out = model(X)
    # err = (out.data.max(1)[1] != y.data).float().sum()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std