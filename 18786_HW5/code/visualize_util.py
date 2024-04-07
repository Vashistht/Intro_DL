import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_img_batch(image_batch): # shape of the image batch should be (N, H, W)
    img_num, H, W = image_batch.shape
    vis_w = 10
    vis_h = int(np.ceil(img_num/10))

    padded_img_num = vis_w*vis_h
    
    if padded_img_num != img_num:
        empty_img_num = padded_img_num - img_num
        padded_image_batch = np.concatenate([image_batch, np.ones( (empty_img_num, H, W))], axis = 0 )
    else:
        padded_image_batch = image_batch
    
    padded_image_batch = np.stack([np.pad(img,(1,1), 'constant', constant_values=(1,1)) for img in padded_image_batch], axis = 0) 
    padded_image_batch = np.reshape(padded_image_batch, (vis_h, vis_w, H+2, W+2 ))
    padded_image_batch = np.concatenate( padded_image_batch, axis=1)
    padded_image_batch = np.concatenate( padded_image_batch, axis=1)
    
    fig = plt.figure(figsize=(vis_w,vis_h))
    plt.imshow(padded_image_batch, cmap='gray')
    plt.axis('off')

def plot_latent_space_images(decoder, device, zi_range = np.arange(-5,5), zj_range = np.arange(5,-5,-1)):
    zi, zj = np.meshgrid(zi_range,zj_range)
    z = np.stack( [ zi.reshape(-1), zj.reshape(-1)], axis = 1)
    z = torch.Tensor(z).to(device)
    x_hat = decoder(z).detach().cpu().numpy().squeeze()
    x_hat = np.reshape(x_hat, (-1,28,28))
    visualize_img_batch(x_hat)

def plot_latent_space(encoder, dataloader, device, num_batches=100):    
    for i in range(num_batches):
        x, y = next(iter(dataloader))
        z = encoder(x.to(device)).detach().cpu().numpy()
        y = y.numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
    plt.colorbar()

def one_hot(labels, class_size):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets