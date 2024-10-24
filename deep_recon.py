
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import astra
from tqdm import tqdm
import skimage
import torch
import torch.nn as nn
from model.resnet import ResNet, ResNet_Grad
from model.data import generate_train
from utils.sinogram import create_sinogram,BP_recon, explicit_radon
from utils.plotting import f_subplots

f = np.load("SLPhan.npy")
f_small = skimage.transform.rescale(f,scale=0.5)
f = torch.as_tensor(f_small).unsqueeze(0).unsqueeze(0).float()
print(f.shape)
#encoder = MiniResBlock2()
#encoder2 = ResBlock2()
decoder = ResNet()
decoder.forward(f)

v,h = 64,64
n_angles = 45
max_angle = np.pi
n_samples = 120
theta = 10000
train_dataset = []
train_sinograms = []
generate_train(n_angles,max_angle,n_samples,theta,v,h,plot=True)

v,h = 64,64
n_angles = 45
max_angle = np.pi
n_samples = 120
theta = 10000
train_dataset = []
train_sinograms = []
for _ in tqdm(range(8000)):
    theta = np.random.randint(100,10000)
    data_pair,g = generate_train(n_angles,max_angle,n_samples,theta,v,h)
    train_dataset.append(data_pair)
    train_sinograms.append(g)

val_dataset = []
for i in tqdm(range(2000)):
    theta = np.random.randint(100,10000)
    data_pair,g_val = generate_train(n_angles,max_angle,n_samples,theta,v,h)
    val_dataset.append(data_pair)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

model = ResNet()
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

train_loss_arr,val_loss_arr = [], []
num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        targets,inputs = data
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_loss_arr.append(avg_train_loss)

    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad(): 
        for data in val_loader:
            targets,inputs = data
            f_recon = model(inputs.float())
            loss = criterion(f_recon, targets)
            val_loss += loss.item()

            f_recon = f_recon.squeeze()
            f_BP = inputs.squeeze()
    avg_val_loss = val_loss / len(val_loader)
    val_loss_arr.append(avg_val_loss)
plt.figure()
plt.plot(train_loss_arr, label='Training Loss')
plt.plot(val_loss_arr, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

ftrue = np.load('SLphan.npy')
f_small = skimage.transform.rescale(ftrue,scale=1)
v_large,h_large=128,128
v_small,h_small = f_small.shape
test_dataset = []
for i in range(10):
    vol_geom, proj_geom, projector_id, g_id, g = create_sinogram(f_small, n_angles, max_angle, n_samples, v_large, h_large)
    gNoisy = astra.functions.add_noise_to_sino(g, np.random.randint(8000,10000))
    gNoisy_id = astra.data2d.create('-sino', proj_geom, gNoisy)
    f_BP = BP_recon(vol_geom, projector_id, gNoisy_id, 'FBP')
    ftrue = torch.as_tensor(ftrue).unsqueeze(0)
    f_BP = torch.as_tensor(f_BP).unsqueeze(0)  
    test_dataset.append((f_BP,f_small))
    
test_loader = DataLoader(test_dataset)
model.eval()
preds = []
with torch.no_grad(): 
    for i in test_loader:
        sample,target = i
        pred = model(sample.float())
        preds.append(pred)
f_subplots(test_dataset[0][0].squeeze(),test_dataset[0][1].squeeze(),np.array(preds[0].squeeze()),'')

ftrue = np.load('SLphan.npy')
f_small = skimage.transform.rescale(ftrue,scale=1)
v_large,h_large=128,128
v_small,h_small = f_small.shape
test_dataset = []
for i in range(10):
    vol_geom, proj_geom, projector_id, g_id, g = create_sinogram(f_small, n_angles, max_angle, n_samples, v_large, h_large)
    gNoisy = astra.functions.add_noise_to_sino(g, np.random.randint(8000,10000))
    gNoisy_id = astra.data2d.create('-sino', proj_geom, gNoisy)
    f_BP = BP_recon(vol_geom, projector_id, gNoisy_id, 'FBP')
    ftrue = torch.as_tensor(ftrue).unsqueeze(0)
    f_BP = torch.as_tensor(f_BP).unsqueeze(0)  
    test_dataset.append((f_BP,f_small))
    
test_loader = DataLoader(test_dataset)
model.eval()
preds = []
with torch.no_grad(): 
    for i in test_loader:
        sample,target = i
        pred = model(sample.float())
        preds.append(pred)
f_subplots(test_dataset[0][0].squeeze(),test_dataset[0][1].squeeze(),np.array(preds[0].squeeze()),'')

v,h = 64,64
n_angles = 45
max_angle = np.pi
n_samples = 120
theta = 10000
train_dataset = []
train_sinograms = []
for _ in tqdm(range(2000)):
    theta = np.random.randint(100,10000)
    inputs,targets = generate_train(n_angles,max_angle,n_samples,theta,v,h,grad=True)
    train_dataset.append((inputs,targets))


val_dataset = []
for i in tqdm(range(500)):
    theta = np.random.randint(100,10000)
    inputs,targets = generate_train(n_angles,max_angle,n_samples,theta,v,h,grad=True)
    val_dataset.append((inputs,targets))

A = explicit_radon(n_angles,max_angle,n_samples,v,h)
A = torch.tensor(A)

train_loader = DataLoader(train_dataset[:500], batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset[:200], batch_size=16, shuffle=True)

model = ResNet_Grad(A)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

train_loss_arr,val_loss_arr = [], []
num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs,targets = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    train_loss_arr.append(avg_train_loss)

    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad(): 
        for data in val_loader:
            inputs,targets = data
            f_recon = model(inputs)
            loss = criterion(f_recon, targets)
            val_loss += loss.item()

            f_recon = f_recon.squeeze()
            f_BP = inputs.squeeze()
    avg_val_loss = val_loss / len(val_loader)
    val_loss_arr.append(avg_val_loss)
plt.figure()
plt.plot(train_loss_arr, label='Training Loss')
plt.plot(val_loss_arr, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
