'''
Contains loss functions. UNET model described here was determined to be sub-optimal. We used the parametric relu one in unet_pbc_prelu.py instead
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    '''double convolution with batch normalization and ReLU activation'''
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_convolution = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=True)
            concatenate_skip_connection = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concatenate_skip_connection)

        return self.final_convolution(x)

class CombinedDiceMSELoss(nn.Module):
    def __init__(self, dice_weight=1.0, mse_weight=1.0):
        super(CombinedDiceMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.dice_weight = dice_weight
        self.mse_weight = mse_weight

    def dice_loss(self, pred, target, smooth=1.0):
        ''' Calculate dice loss for each sample, averaged across the batch '''
        #sigmoid inputs because pred inputs is logits
        pred = torch.sigmoid(pred)

        #flatten label and prediction tensors along spatial dimensions, keep batch dimension
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        dice_loss = 1 - ((2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth))

        return dice_loss.mean()

    def forward(self, pred, target):
        # Assuming pred and target are of shape [batch_size, 2, 256, 256]
        # Channel 0: segmentation, Channel 1: scalar field

        dice_loss = self.dice_loss(pred[:, 0, :, :], target[:, 0, :, :]) #pred inputs is logits
        mse_loss = self.mse(pred[:, 1, :, :], target[:, 1, :, :])

        weighted_dice_loss = self.dice_weight * dice_loss
        weighted_mse_loss = self.mse_weight * mse_loss

        # Combine the two losses with weights
        loss = weighted_dice_loss + weighted_mse_loss
        return loss, weighted_dice_loss, weighted_mse_loss
    
class BCE_MSE_loss(nn.Module):
    def __init__(self, bce_weight=1.0, mse_weight=1.0):
        super(BCE_MSE_loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight

    def forward(self, pred, target):
        # Assuming pred and target are of shape [batch_size, 2, 256, 256]
        # Channel 0: segmentation, Channel 1: scalar field

        bce_loss = self.bce(pred[:, 0, :, :], target[:, 0, :, :])
        mse_loss = self.mse(pred[:, 1, :, :], target[:, 1, :, :])

        weighted_bce_loss = self.bce_weight * bce_loss
        weighted_mse_loss = self.mse_weight * mse_loss

        # Combine the two losses with scaling factors
        loss = weighted_bce_loss + weighted_mse_loss
        return loss, weighted_bce_loss, weighted_mse_loss
    
class Dice_MSE_sseCELL_sseVEGF_loss(nn.Module):
    def __init__(self, dice_weight=1.0, mse_weight=1.0, sseCELL_weight=1.0, sseVEGF_weight=1.0):
        super(Dice_MSE_sseCELL_sseVEGF_loss, self).__init__()
        self.mse = nn.MSELoss()
        self.dice_weight = dice_weight
        self.mse_weight = mse_weight
        self.sseCELL_weight = sseCELL_weight
        self.sseVEGF_weight = sseVEGF_weight

    def dice_loss(self, pred, target, smooth=1.0):
        ''' Calculate dice loss for each sample, averaged across the batch '''
        #sigmoid inputs because pred inputs is logits
        pred = torch.sigmoid(pred)

        #flatten label and prediction tensors along spatial dimensions, keep batch dimension
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        dice_loss = 1 - ((2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth))

        return dice_loss.mean()

    def forward(self, pred, target):
        # Assuming pred and target are of shape [batch_size, 2, 256, 256]
        # Channel 0: segmentation, Channel 1: scalar field

        # batch_size = pred.size(0)

        dice_loss = self.dice_loss(pred[:, 0, :, :], target[:, 0, :, :]) #pred inputs is logits
        mse_loss = self.mse(pred[:, 1, :, :], target[:, 1, :, :])

        # sseCELL_loss = torch.mean((torch.sum(pred[:, 0, :, :], dim=[1, 2]) - torch.sum(target[:, 0, :, :], dim=[1, 2])) ** 2) #constraint on total area of cells
        sseVEGF_loss = torch.mean((torch.sum(pred[:, 1, :, :], dim=[1, 2]) - torch.sum(target[:, 1, :, :], dim=[1, 2])) ** 2) #constraint on total area of cells


        # cell and vegf constraints absolute error
        sigmoid_pred = torch.sigmoid(pred[:, 0, :, :])
        thresholded_pred = torch.where(sigmoid_pred > 0.5, 1, 0)
        sigmoid_target = torch.sigmoid(target[:, 0, :, :])
        thresholded_target = torch.where(sigmoid_target > 0.5, 1, 0)

        absCELL_loss = torch.mean(torch.abs(torch.sum(thresholded_pred, dim=[1, 2]) - torch.sum(thresholded_target, dim=[1, 2])))

        absVEGF_loss = torch.mean(torch.abs(torch.sum(pred[:, 1, :, :], dim=[1, 2]) - torch.sum(target[:, 1, :, :], dim=[1, 2])))
        # absVEGF_loss = torch.mean((torch.mean(pred[:, 1, :, :], dim=[1, 2]) - torch.mean(target[:, 1, :, :], dim=[1, 2]))**2)

        weighted_dice_loss = self.dice_weight * dice_loss
        weighted_mse_loss = self.mse_weight * mse_loss
        weighted_sseCELL_loss = self.sseCELL_weight * absCELL_loss
        weighted_sseVEGF_loss = self.sseVEGF_weight * absVEGF_loss

        # Combine the two losses with weights
        loss = weighted_dice_loss + weighted_mse_loss + weighted_sseCELL_loss + weighted_sseVEGF_loss
        return loss, weighted_dice_loss, weighted_mse_loss, weighted_sseCELL_loss, weighted_sseVEGF_loss
    
class BCE_MSE_sseCELL_sseVEGF_loss(nn.Module):
    def __init__(self, bce_weight=1.0, mse_weight=1.0, sseCELL_weight=1.0, sseVEGF_weight=1.0):
        super(BCE_MSE_sseCELL_sseVEGF_loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.sseCELL_weight = sseCELL_weight
        self.sseVEGF_weight = sseVEGF_weight

    def forward(self, pred, target):
        # Assuming pred and target are of shape [batch_size, channel, 256, 256]
        # Channel 0: segmentation, Channel 1: scalar field

        # batch_size = pred.size(0)

        bce_loss = self.bce(pred[:, 0, :, :], target[:, 0, :, :])
        mse_loss = self.mse(pred[:, 1, :, :], target[:, 1, :, :])
        
        # cell and vegf constraints absolute error
        sigmoid_scalar = 1
        sigmoid_pred = torch.sigmoid(pred[:, 0, :, :]*sigmoid_scalar)
        # thresholded_pred = torch.where(sigmoid_pred > 0.5, 1, 0)
        sigmoid_target = torch.sigmoid(target[:, 0, :, :]*sigmoid_scalar)
        # thresholded_target = torch.where(sigmoid_target > 0.5, 1, 0)

        absCELL_loss = torch.mean(torch.abs(torch.sum(sigmoid_pred.float(), dim=[1, 2]) - torch.sum(sigmoid_target.float(), dim=[1, 2])))
        absVEGF_loss = torch.mean(torch.abs(torch.sum(pred[:, 1, :, :], dim=[1, 2]) - torch.sum(target[:, 1, :, :], dim=[1, 2])))
        
        weighted_bce_loss = self.bce_weight * bce_loss
        weighted_mse_loss = self.mse_weight * mse_loss
        weighted_sseCELL_loss = self.sseCELL_weight * absCELL_loss
        weighted_sseVEGF_loss = self.sseVEGF_weight * absVEGF_loss

        # Combine the two losses with scaling factors
        loss = weighted_bce_loss + weighted_mse_loss + weighted_sseCELL_loss + weighted_sseVEGF_loss
        return loss, weighted_bce_loss, weighted_mse_loss, weighted_sseCELL_loss, weighted_sseVEGF_loss


#3 loss terms
class Dice_MSE_sseVEGF_loss(nn.Module):
    def __init__(self, dice_weight=1.0, mse_weight=1.0, sseVEGF_weight=1.0):
        super(Dice_MSE_sseVEGF_loss, self).__init__()
        self.mse = nn.MSELoss()
        self.dice_weight = dice_weight
        self.mse_weight = mse_weight
        self.sseVEGF_weight = sseVEGF_weight

    def dice_loss(self, pred, target, smooth=1.0):
        ''' Calculate dice loss for each sample, averaged across the batch '''
        #sigmoid inputs because pred inputs is logits
        pred = torch.sigmoid(pred)

        #flatten label and prediction tensors along spatial dimensions, keep batch dimension
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        dice_loss = 1 - ((2.0 * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth))

        return dice_loss.mean()

    def forward(self, pred, target):
        # Assuming pred and target are of shape [batch_size, 2, 256, 256]
        # Channel 0: segmentation, Channel 1: scalar field

        # batch_size = pred.size(0)

        dice_loss = self.dice_loss(pred[:, 0, :, :], target[:, 0, :, :]) #pred inputs is logits
        mse_loss = self.mse(pred[:, 1, :, :], target[:, 1, :, :])

        # sseCELL_loss = torch.mean((torch.sum(pred[:, 0, :, :], dim=[1, 2]) - torch.sum(target[:, 0, :, :], dim=[1, 2])) ** 2) #constraint on total area of cells
        sseVEGF_loss = torch.mean((torch.sum(pred[:, 1, :, :], dim=[1, 2]) - torch.sum(target[:, 1, :, :], dim=[1, 2])) ** 2) #constraint on total area of cells


        # vegf constraint absolute error

        absVEGF_loss = torch.mean(torch.abs(torch.sum(pred[:, 1, :, :], dim=[1, 2]) - torch.sum(target[:, 1, :, :], dim=[1, 2])))
        # absVEGF_loss = torch.mean((torch.mean(pred[:, 1, :, :], dim=[1, 2]) - torch.mean(target[:, 1, :, :], dim=[1, 2]))**2)

        weighted_dice_loss = self.dice_weight * dice_loss
        weighted_mse_loss = self.mse_weight * mse_loss
        weighted_sseVEGF_loss = self.sseVEGF_weight * absVEGF_loss

        # Combine the two losses with weights
        loss = weighted_dice_loss + weighted_mse_loss + weighted_sseVEGF_loss
        return loss, weighted_dice_loss, weighted_mse_loss, weighted_sseVEGF_loss
    

class BCE_MSE_sseVEGF_loss(nn.Module):
    def __init__(self, bce_weight=1.0, mse_weight=1.0, sseVEGF_weight=1.0):
        super(BCE_MSE_sseVEGF_loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight
        self.sseVEGF_weight = sseVEGF_weight

    def forward(self, pred, target):
        # Assuming pred and target are of shape [batch_size, channel, 256, 256]
        # Channel 0: segmentation, Channel 1: scalar field

        # batch_size = pred.size(0)

        bce_loss = self.bce(pred[:, 0, :, :], target[:, 0, :, :])
        mse_loss = self.mse(pred[:, 1, :, :], target[:, 1, :, :])
        
        # vegf constraint absolute error
        absVEGF_loss = torch.mean(torch.abs(torch.sum(pred[:, 1, :, :], dim=[1, 2]) - torch.sum(target[:, 1, :, :], dim=[1, 2])))
        
        weighted_bce_loss = self.bce_weight * bce_loss
        weighted_mse_loss = self.mse_weight * mse_loss
        weighted_sseVEGF_loss = self.sseVEGF_weight * absVEGF_loss

        # Combine the two losses with scaling factors
        loss = weighted_bce_loss + weighted_mse_loss + weighted_sseVEGF_loss
        return loss, weighted_bce_loss, weighted_mse_loss, weighted_sseVEGF_loss