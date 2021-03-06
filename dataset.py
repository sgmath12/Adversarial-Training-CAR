import torchvision
import torchvision.transforms as transforms
import pdb

def MNIST(root = '/dataset',normalize = False,download = False):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = torchvision.datasets.MNIST(root = root, train=True, transform=transform, target_transform=None, download=download)
    val_set = torchvision.datasets.MNIST(root = root, train=False, transform=transform, target_transform=None, download=download)

    return train_set,val_set


def FashionMNIST(root = '/dataset',normalize = False):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = torchvision.datasets.MNIST(root = root, train=True, transform=transform, target_transform=None, download=True)
    val_set = torchvision.datasets.MNIST(root = root, train=False, transform=transform, target_transform=None, download=True)

    return train_set,val_set

def CIFAR10(root = '/dataset',normalize = False,download = False):
    if normalize == True:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    train_set = torchvision.datasets.CIFAR10(root = root, train=True, transform = train_transform, target_transform=None, download=download)
    val_set = torchvision.datasets.CIFAR10(root = root , train=False, transform = test_transform, target_transform=None, download=download)
    
    return train_set,val_set

def CIFAR100(root = '/dataset',normalize = False,download = False):
    if normalize == True:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    train_set = torchvision.datasets.CIFAR100(root = root, train=True, transform = train_transform, target_transform=None, download=download)
    val_set = torchvision.datasets.CIFAR100(root = root , train=False, transform = test_transform, target_transform=None, download=download)
    
    return train_set,val_set

def IMAGENET(root = '/dataset',normalize = False):
    if normalize == False:
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        
    else:
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    # train_set = torchvision.datasets.ImageFolder(root + 'train', transform = transform)
    val_set = torchvision.datasets.ImageFolder(root + 'val', transform = transform)

    # return train_set,val_set
    return None,val_set


def SVHN(root = '/dataset',normalize = False,download = False):
    if normalize == True:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    train_set = torchvision.datasets.SVHN(root = root, split='train', transform = train_transform, target_transform=None, download=download)
    val_set = torchvision.datasets.SVHN(root = root , split='test', transform = test_transform, target_transform=None, download=download)
    
    return train_set,val_set