import torch
import torch.nn as nn
import torch.nn.functional as f
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
from scipy.stats import gaussian_kde
from skimage.restoration import denoise_wavelet
from xgboost import XGBRegressor
from torchmetrics.functional import peak_signal_noise_ratio as PSNR



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Importing Images ...')
high = []
i = 0
dir = 'our485/high'
for img in sorted(os.listdir('our485/high')):
    if img.endswith('.png'):
        x = Image.open(dir + '/' + img)
        x = torch.from_numpy(np.array(x))
        high.append(x)
        i += 1
        print(f'Count : {i}', end = '\r')
high = torch.stack(high).permute(0, 3, 1, 2) / 255

low = []
i = 0
dir = 'our485/low'
for img in sorted(os.listdir('our485/low')):
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

print('Training Preprocessing Model ...')
input_x  = np.array([np.histogram(img, bins=256, range=(0,1))[0] for img in low.reshape(-1, 400, 600)])
shape = (low.shape[0]*3, low.shape[2], low.shape[3])
ys = []

for imhigh in high.reshape(shape):
    quantiles = np.percentile(imhigh, np.linspace(0, 100, 21))
    ys.append(quantiles)
ys = np.array(ys)

model_xgb = XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.2, tree_method='hist')
model_xgb.fit(input_x, ys)
model_xgb.save_model('models/my_xgb_model.model')

out = model_xgb.predict(np.array([np.histogram(img, bins=256, range=(0,1))[0] for img in low.reshape(-1, 400, 600)]))
print('Preprocessing Images ...')
new_input = []
for i in range(len(low)):
    im=[]
    hist = hist_from_quant(out[3*i])
    im.append(convert(low[i][0], hist))
    hist = hist_from_quant(out[3*i+1])
    im.append(convert(low[i][1], hist))
    hist = hist_from_quant(out[3*i+2])
    im.append(convert(low[i][2], hist))
    im = denoise(im)
    new_input.append(im)

new_input = torch.from_numpy(np.array(new_input)).float()

batch_size = 8

def get_batch():
    idx = torch.randint(0, len(high), (batch_size, ))
    x, y = new_input[idx], high[idx]
    return x, y
print('Training Generation Model ...')
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
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
epochs = 200
l = []
PATH = "models/transformer_conv_transform_new_input.pt"
model.train()
for epoch in range(epochs):
    lossy = 0
    print(f'Epoch {epoch + 1} : ')
    mse = 0
    psnr = 0
    for j in tqdm(range(485 // batch_size)):
        x, y = get_batch()
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss1 = -PSNR(y_hat, y)
        loss2 = f.l1_loss(y_hat, y)
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mse += loss2.item()
        psnr += loss1.item()
        del x, y, y_hat
        lossy += loss.item()
    print('Loss :',lossy / (485 // batch_size), 'PSNR :', psnr / (485 // batch_size))
    EPOCH = epoch + 1
    torch.save({
            'epoch': EPOCH, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': round(loss.item(), 2),
            }, PATH)

print('Training Completed !')
