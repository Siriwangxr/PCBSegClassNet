import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        torch.sum(y_true_f * y_true_f) + torch.sum(y_pred_f * y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return torch.mean(1.0 - dice_coef(y_true, y_pred))


def jacard_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (
        torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection + smooth
    )


def jacard_coef_loss(y_true, y_pred):
    return torch.mean(1.0 - jacard_coef(y_true, y_pred))


def gaussian_kernel(window_size, sigma):
    x = torch.arange(window_size).float() - window_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum()
def ssim_loss(y_true, y_pred, window_size=11, sigma=1.5, c1=0.01**2, c2=0.03**2):
    # Create a 2D Gaussian kernel
    kernel = gaussian_kernel(window_size, sigma).unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(y_true.size(1), 1, window_size, window_size).to(y_true.device)

    # Compute the mean and variance of y_true and y_pred
    mu1 = F.conv2d(y_true, kernel, padding=window_size//2, groups=y_true.size(1))
    mu2 = F.conv2d(y_pred, kernel, padding=window_size//2, groups=y_pred.size(1))
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(y_true*y_true, kernel, padding=window_size//2, groups=y_true.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(y_pred*y_pred, kernel, padding=window_size//2, groups=y_pred.size(1)) - mu2_sq
    sigma12 = F.conv2d(y_true*y_pred, kernel, padding=window_size//2, groups=y_true.size(1)) - mu1_mu2

    # Compute SSIM
    ssim_map = ((2*mu1_mu2 + c1)*(2*sigma12 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))


    return 1 - ssim_map.mean()


def DISLoss(y_true, y_pred):
    return (
        dice_loss(y_true, y_pred)
        + jacard_coef_loss(y_true, y_pred)
        + ssim_loss(y_true, y_pred)
    )