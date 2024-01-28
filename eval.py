import sys
import torch
import torch.nn as nn
import torch.nn.functional as f
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
import cv2
import time
import argparse
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from skimage.restoration import denoise_wavelet
from xgboost import XGBRegressor

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', help = 'Directory of output images')
parser.add_argument('-i', '--input', help = 'Directory of input images')
args = parser.parse_args()

path_to_test = args.input
path_to_result = args.output

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device == torch.device('cuda') :
    print('GPU Found !')
print('Loading trained models ...')
print()
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model = 300, nhead = 10, dim_feedforward = 512, batch_first = True)
        self.transformer = nn.TransformerEncoder(self.layer, num_layers = 2)
        self.extra = nn.Sequential(
            nn.Conv2d(3, 128, (3, 3), padding = 'same'),
            nn.GELU(),
            nn.Conv2d(128, 3, (3, 3), padding = 'same')
        )
    def forward(self, x):
        x = self.img_to_patch(x, 10)
        x = self.transformer(x)
        x = self.patch_to_img(x, 10, 3, 400, 600)
        x = self.extra(x)
        return f.sigmoid(x)
    def img_to_patch(self, x, patch_size):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.flatten(1,2)            
        x = x.flatten(2,4)   
        return x
    def patch_to_img(self, x, patch_size, C, H, W):
        x = x.view(-1, H*W//(patch_size)**2, C , patch_size , patch_size)
        x = x.view(-1, H // patch_size, W // patch_size, C, patch_size, patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x = x.reshape(-1, C, H, W)
        return x
    
model = Model()
model = model.to(device)
checkpoint = torch.load("models/transformer_conv_transform_new_input.pt", map_location = device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

model_xgb = XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.2)
model_xgb.load_model('models/my_xgb_model.model')

print('Loaded trained models ...')
print()
print('Importing images ...')
print()
low = []
i = 0
dir = path_to_test
for img in sorted(os.listdir(dir)):
    if img.endswith('.png'):
        x = Image.open(dir + '/' + img)
        x = torch.from_numpy(np.array(x))
        low.append(x)
        i += 1
        print(f'Count : {i}', end = '\r')
low = torch.stack(low).permute(0, 3, 1, 2) / 255


def hist_from_quant(quant, c=0.75):
    x = np.linspace(0, 1, 255)
    kde = gaussian_kde(quant, c*quant.std())
    pdf = kde(x)
    return pdf

def convert(img, hist):
    input_hist, _ = np.histogram(img, bins=256, range=(0, 1))
    input_hist = input_hist / np.sum(input_hist)
    desired_hist = hist / np.sum(hist)
    input_cumsum = np.cumsum(input_hist)
    desired_cumsum = np.cumsum(desired_hist)
    mapping_func = np.interp(input_cumsum, desired_cumsum, np.linspace(0, 1, 255))
    matched_image = np.interp(img, np.linspace(0, 1, 256), mapping_func)
    return matched_image
print('Preprocessing Images ...')
start = time.time()
out = model_xgb.predict(np.array([np.histogram(img, bins=256, range=(0,1))[0] for img in low.reshape(-1, 400, 600)]))
new_input = []
for i in range(len(low)):
    im=[]
    hist = hist_from_quant(out[3*i])
    im.append(convert(low[i][0], hist))
    hist = hist_from_quant(out[3*i+1])
    im.append(convert(low[i][1], hist))
    hist = hist_from_quant(out[3*i+2])
    im.append(convert(low[i][2], hist))
    new_input.append(im)

new_input = torch.from_numpy(np.array(new_input)).float()
print(f'Time taken : {round(time.time() - start, 3)}')
print()
print('Generating new images ...')
start = time.time()
with torch.no_grad():
    result = model(new_input.to(device))
for i in range(len(result)):
    cv2.imwrite(path_to_result + '/' + f'result{i}.png', cv2.cvtColor(result[i].permute(1, 2, 0).numpy() * 255, cv2.COLOR_RGB2BGR))
print(f'Time taken : {round(time.time() - start, 3)}')
print(f'Generated images saved to {path_to_result}')
