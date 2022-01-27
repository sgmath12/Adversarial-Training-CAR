from torch import nn
import torch
import pdb



class FeatureExtractor(nn.Module):
    def __init__(self, model,layer_idx):
        '''
        model : torch model
        ex)
            features = FeatureExtractor(vgg16)
            z = features(x)            # you should forward input x to get features.
            features.activations[0]    # torch tensor [batch size, channel, height, width].      
            features.layer_numbers     # the number of total registered layers.
            features.names[0]          # ex) Conv2d, ReLU, BatchNorm2d.
        '''
        super(FeatureExtractor,self).__init__()
        self.model = model
        self.activations = {}
        self.names = {}
        self.layer_numbers = 0
        self.layer_idx = layer_idx
        
        self.non_robust_indexes = None
        self.target_layer_name = None
        self.register_forward_hook(model)
        self.target_layer_name = None
        self.registerd = None
        

    def register_forward_hook(self,model):
        idx = 0
        for name,layer in model.named_modules():
            # if isinstance(layer, nn.Conv2d) or isinstance(layer,nn.ReLU) or isinstance(layer,nn.BatchNorm2d) :
            # if isinstance(layer, nn.ReLU):
            if isinstance(layer, nn.Conv2d):
                layer.register_forward_hook(self.get_activation(idx))
                self.names[idx] = name
                idx += 1
        self.layer_numbers = idx

    def register_forward_pre_hook(self,model,target_layer_idx):
        for name,layer in model.named_modules():
            self.target_layer_name = name
            if isinstance(layer, nn.Conv2d) or isinstance(layer,nn.ReLU) or isinstance(layer,nn.BatchNorm2d) :
                if idx == target_layer_idx:
                    self.registerd = layer.register_forward_pre_hook(self.modify_activations())
                    break
                idx += 1

    def unregister_forward_pre_hook(self,model,target_layer_idx):
        idx = 0
        self.registerd.remove()

    def get_activation(self,idx):
        def hook_fn(module,input,output):
            # self.activations[idx] = output
            self.activations[idx] = input[0]
        return hook_fn

    def modify_activations(self):
        def hook_fn(module,input):
            gamma = 5
            # feature = input[0][:,self.non_robust_indexes,:,:]
            first_input, *rest_input = input
            return (first_input * (self.non_robust_indexes) * (1 + gamma) + first_input * (~self.non_robust_indexes), *rest_input)

            # print ("modified")
        return hook_fn
            
    def forward(self, x):
        return self.model(x)