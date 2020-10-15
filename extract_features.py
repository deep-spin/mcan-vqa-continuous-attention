import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from torch.autograd import gradcheck, Variable
import pickle
import os
from torchvision import transforms
from PIL import Image

class AttentiveCNN(nn.Module):
    def __init__(self):
        super(AttentiveCNN, self).__init__()
        
        # ResNet-152 backend
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-2] # delete the last fc layer and avg pool.
        resnet_conv = nn.Sequential(*modules) # last conv feature
        
        self.resnet_conv = resnet_conv

    def forward(self, images):
        '''
        Input: images
        Output: V=[v_1, ..., v_n], v_g
        '''
        # Last conv layer feature map
        A = self.resnet_conv(images)
        # V = [ v_1, v_2, ..., v_49 ]
        V = A.view(A.size(0), A.size(1), -1).transpose(1,2)

        return V

transform = transforms.Compose([
        transforms.Resize((448, 448)), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

model=AttentiveCNN().cuda()
model.eval()

i=0
paths=[]
for image_path in os.listdir('./val2014'):
    i+=1
    image = Image.open(os.path.join('./val2014', image_path)).convert('RGB')
    image=transform(image)
    if len(paths)==0:
        images = image.unsqueeze(0)
    else:
        images = torch.cat([images,image.unsqueeze(0)],0)
    paths.append(image_path)
    if i%1000==0:
        print(i)
    if images.size(0)==10:
        v = model(images.cuda())
        for j in range(v.size(0)):
            np.savez('./features/val/'+paths[j].replace('.jpg',''),v[j].clone().detach().cpu().numpy())
        paths=[]
        del(images)
        del(v)
v = model(images.cuda())
for j in range(v.size(0)):
    np.savez('./features/val/'+paths[j].replace('.jpg',''),v[j].clone().detach().cpu().numpy())
