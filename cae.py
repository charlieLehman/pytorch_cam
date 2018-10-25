import torch
from torch import nn
import numpy as np
from matplotlib.colors import hsv_to_rgb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, final_activation=True, activation=nn.ReLU):
        super(Up ,self).__init__()

        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.residual = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, filter_size, padding=1),
            nn.BatchNorm2d(out_channels),
            activation(),
            )
        self.features = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, filter_size, padding=1),
            nn.BatchNorm2d(out_channels),
            activation(),
            nn.ConvTranspose2d(out_channels, out_channels, filter_size, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.final_activation=final_activation
        self.activation = activation()
        self.__initialize_weights()

    def __initialize_weights(self):
        for seq in [self.features, self.residual]:
            for layer in seq:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)

    def forward(self, x, pool_index, pool_size):
        x = self.unpool(x, pool_index, pool_size)
        x = self.residual(x)
        x = self.features(x) + x
        if self.final_activation:
            x = self.activation(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, activation=nn.ReLU):
        super(Down ,self).__init__()

        self.pool_index = None
        self.pool_size = None
        self.pool = nn.Sequential(
            activation(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            )

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, filter_size, padding=1),
            nn.BatchNorm2d(out_channels),
            activation(),
            )

        self.features = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, filter_size, padding=1),
            nn.BatchNorm2d(out_channels),
            activation(),
            nn.Conv2d(out_channels, out_channels, filter_size, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.__initialize_weights()

    def __initialize_weights(self):
        for seq in [self.features, self.pool, self.residual]:
            for layer in seq:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        self.pool_size = x.size()
        x = self.residual(x)
        x = self.features(x) + x
        x, idx = self.pool(x)
        self.pool_index = idx
        return x

    def plot_kernels(self, num_cols=6):
        weights = self.features[0].weight.data.cpu()
        weights = np.transpose(weights, [0,2,3,1])
        print(weights.shape, weights.max(), weights.mean(), weights.min())
        num_kernels = weights.shape[0]* weights.shape[-1]
        num_rows = 1 + num_kernels // num_cols
        fig = plt.figure(figsize=(num_cols,num_rows))
        for j in range(weights.shape[-1]):
            for i in range(weights.shape[0]):
                ax1 = fig.add_subplot(num_rows,num_cols,j+i+1)
                ax1.imshow(weights[i,:,:,j])
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig('stl10_test_embed_cae.png')

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size):
        super(Bottleneck ,self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, filter_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, in_channels, 1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        self.__initialize_weights()

    def __initialize_weights(self):
        for seq in [self.features]:
            for layer in seq:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = self.features(x)
        out = x + out
        out = nn.functional.relu(out)
        return x

class Attention(nn.Module):
    def __init__(self, h, w):
        super(Attention, self).__init__()

        x = torch.linspace(-1,1,w).repeat(h,1)
        y = torch.linspace(-1,1,h).repeat(w,1).t()
        self.field = nn.Parameter(torch.stack([x,y]).unsqueeze(0).unsqueeze(2).unsqueeze(2), requires_grad=False)

    def forward(self, x, centers, betas, psi):
        dist = (centers-self.field).pow(2).sum(1)
        theta = dist.div(2*betas)
        attn = (nn.functional.softmax(psi, dim=1)/torch.sqrt(2*np.pi*betas)).mul(torch.exp(-theta)).sum(2)
        return attn

class GMM(nn.Module):
    def __init__(self):
        super(GMM, self).__init__()

    def forward(self, x, centers, betas, psi):
        dist = (centers-x).pow(2).sum(1)
        theta = dist.div(2*betas)
        attn = (nn.functional.softmax(psi, dim=1)/torch.sqrt(2*np.pi*betas)).mul(torch.exp(-theta)).sum(2)
        return attn

class SegNet_Classifier(nn.Module):
    def __init__(self, n_classes):
        super(SegNet_Classifier, self).__init__()
        self.down1 = Down(3,64,3)
        self.down2 = Down(64,128,3)
        self.down3 = Down(128,256,3)
        self.up3 = Up(256,128,3)
        self.up2 = Up(128,64,3)
        self.up1 = Up(64,n_classes,3)
        self.attn = Up(64,1,3)
        self.n_classes = n_classes

    def pixelwise_classifier(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.up3(x, self.down3.pool_index, self.down3.pool_size)
        x1 = self.up2(x, self.down2.pool_index, self.down2.pool_size)
        x = self.up1(x1, self.down1.pool_index, self.down1.pool_size)
        attn = self.attn(x1, self.down1.pool_index, self.down1.pool_size)
        return x.mul(attn), attn

    def segment(self, x):
        gs_im = x.mean(1)
        gs_mean = gs_im.mean()
        gs_min = gs_im.min()
        gs_max = torch.max((gs_im-gs_min))
        gs_im = (gs_im - gs_min)/gs_max
        x, attn = self.pixelwise_classifier(x)
        hue = (torch.argmax(x, dim=1).float() + 0.5)/self.n_classes
        pred = torch.argmax(x, dim=1)
        im_lvl_prob = nn.functional.softmax(x.mean(-1).mean(-1), dim=1)
        im_lvl_conf = torch.max(im_lvl_prob, dim=1)[0]
        im_lvl_pred_ = torch.argmax(im_lvl_prob, dim=1).unsqueeze(-1).unsqueeze(-1)
        conf = torch.max(nn.functional.softmax(x, dim=1), dim=1)[0]
        hsv_im = torch.stack((hue.float(), conf.float(), gs_im.float()), -1)

        class_selected_seg = []
        for n in range(self.n_classes):
            conf_ = where(pred==n, conf, 0)
            hsv_im_ = torch.stack((hue.float(), conf_.float(), gs_im.float()), -1)
            class_selected_seg.append(hsv_im_)
        return hsv_im, im_lvl_conf.squeeze(), im_lvl_pred_.squeeze(), im_lvl_prob.squeeze(), class_selected_seg, attn

    def forward(self, x):
        x, _ = self.pixelwise_classifier(x)
        x = x.mean(-1).mean(-1)
        return x

class SegNet_VOC_Classifier(nn.Module):
    def __init__(self, n_classes):
        super(SegNet_VOC_Classifier, self).__init__()
        self.down1 = Down(3,16,3)
        self.down2 = Down(16,32,3)
        self.down3 = Down(32,64,3)
        self.down4 = Down(64,128,3)
        self.down5 = Down(128,256,3)
        self.down6 = Down(256,512,3)
        self.up6 = Up(512,256,3)
        self.up5 = Up(256,128,3)
        self.up4 = Up(128,64,3)
        self.up3 = Up(64,32,3)
        self.up2 = Up(32,16,3)
        self.up1 = Up(16,n_classes,3)
        self.attn = Up(16,1,3)
        self.n_classes = n_classes

    def pixelwise_classifier(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.down6(x)
        x = self.up6(x, self.down6.pool_index, self.down6.pool_size)
        x = self.up5(x, self.down5.pool_index, self.down5.pool_size)
        x = self.up4(x, self.down4.pool_index, self.down4.pool_size)
        x = self.up3(x, self.down3.pool_index, self.down3.pool_size)
        x1 = self.up2(x, self.down2.pool_index, self.down2.pool_size)
        x = self.up1(x1, self.down1.pool_index, self.down1.pool_size)
        attn = self.attn(x1, self.down1.pool_index, self.down1.pool_size)
        return x.mul(attn), attn

    def segment(self, x):
        gs_im = x.mean(1)
        gs_mean = gs_im.mean()
        gs_min = gs_im.min()
        gs_max = torch.max((gs_im-gs_min))
        gs_im = (gs_im - gs_min)/gs_max
        x, attn = self.pixelwise_classifier(x)
        hue = (torch.argmax(x, dim=1).float() + 0.5)/self.n_classes
        pred = torch.argmax(x, dim=1)
        im_lvl_prob = nn.functional.softmax(x.mean(-1).mean(-1), dim=1)
        im_lvl_conf = torch.max(im_lvl_prob, dim=1)[0]
        im_lvl_pred_ = torch.argmax(im_lvl_prob, dim=1).unsqueeze(-1).unsqueeze(-1)
        conf = torch.max(nn.functional.softmax(x, dim=1), dim=1)[0]
        hsv_im = torch.stack((hue.float(), conf.float(), gs_im.float()), -1)

        class_selected_seg = []
        for n in range(self.n_classes):
            conf_ = where(pred==n, conf, 0)
            hsv_im_ = torch.stack((hue.float(), conf_.float(), gs_im.float()), -1)
            class_selected_seg.append(hsv_im_)
        return hsv_im, im_lvl_conf.squeeze(), im_lvl_pred_.squeeze(), im_lvl_prob.squeeze(), class_selected_seg, attn

    def forward(self, x):
        x, _ = self.pixelwise_classifier(x)
        x = x.mean(-1).mean(-1)
        return x

class SegNet_Hyperpolar(nn.Module):
    def __init__(self, n_classes, n_centers, embed_dim):
        super(SegNet_Hyperpolar, self).__init__()
        self.down1 = Down(3,64,3)
        self.down2 = Down(64,128,3)
        self.down3 = Down(128,256,3)
        self.up3 = Up(256,128,3)
        self.up2 = Up(128,64,3)
        self.up1 = Up(64,embed_dim,3)
        self.hp = GMM()
        self.get_centers = nn.Linear(32*32*embed_dim, 32*32*2*n_centers)
        self.get_betas = nn.Linear(32*32*embed_dim, 32*32*n_centers)
        self.get_psi = nn.Linear(32*32*embed_dim, 32*32*n_centers)
        self.recon = nn.Conv2d(n_centers, 3, 3)
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.n_centers = n_centers

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.up3(x, self.down3.pool_index, self.down3.pool_size)
        x = self.up2(x, self.down2.pool_index, self.down2.pool_size)
        x = self.up1(x, self.down1.pool_index, self.down1.pool_size)
        x1 = x.view(-1, 32*32*self.embed_dim)
        centers = self.get_centers(x1).view(1, 2, self.n_centers, 32, 32)
        betas = self.get_betas(x1).view(1, self.n_centers, 32, 32)
        psi = self.get_psi(x1).view(1, self.n_centers, 32, 32)
        x = self.hp(x, centers, betas, psi)
        x = self.recon(x)
        return x

    def segment(self, x):
        gs_im = x.mean(1)
        gs_mean = gs_im.mean()
        gs_min = gs_im.min()
        gs_max = torch.max((gs_im-gs_min))
        gs_im = (gs_im - gs_min)/gs_max
        x, attn = self.pixelwise_classifier(x)
        hue = (torch.argmax(x, dim=1).float() + 0.5)/self.n_classes
        pred = torch.argmax(x, dim=1)
        im_lvl_prob = nn.functional.softmax(x.mean(-1).mean(-1), dim=1)
        im_lvl_conf = torch.max(im_lvl_prob, dim=1)[0]
        im_lvl_pred_ = torch.argmax(im_lvl_prob, dim=1).unsqueeze(-1).unsqueeze(-1)
        conf = torch.max(nn.functional.softmax(x, dim=1), dim=1)[0]
        hsv_im = torch.stack((hue.float(), conf.float(), gs_im.float()), -1)

        class_selected_seg = []
        for n in range(self.n_classes):
            conf_ = where(pred==n, conf, 0)
            hsv_im_ = torch.stack((hue.float(), conf_.float(), gs_im.float()), -1)
            class_selected_seg.append(hsv_im_)
        return hsv_im, im_lvl_conf.squeeze(), im_lvl_pred_.squeeze(), im_lvl_prob.squeeze(), class_selected_seg, attn
def where(cond, x_1, x_2):
    cond = cond.float()
    return (cond * x_1) + ((1-cond) * x_2)
