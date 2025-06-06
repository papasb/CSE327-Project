# module/sps.py
import torch  # import pytorch library
import torch.nn as nn  # import pytorch nn module as nn
# import lif node classes from spikingjelly.activation_based.neuron
from spikingjelly.activation_based.neuron import (
    LIFNode,
    ParametricLIFNode   
)
from timm.models.layers import to_2tuple  # import to_2tuple function from timm.models.layers

class MS_SPS(nn.Module):  # define ms_sps class inheriting from nn.module
    def __init__(  # define initializer method
        self,  # self reference
        img_size_h=128,  # image height default 128
        img_size_w=128,  # image width default 128
        patch_size=4,  # patch size default 4
        in_channels=2,  # number of input channels default 2
        embed_dims=256,  # embedding dimensions default 256
        pooling_stat="1111",  # pooling statistic default '1111'
        spike_mode="lif",  # spike mode default 'lif'
    ):  # end parameter list
        super().__init__()  # call parent initializer
        self.image_size = [img_size_h, img_size_w]  # store image size as a list
        patch_size = to_2tuple(patch_size)  # convert patch size to 2-tuple
        self.patch_size = patch_size  # store patch size
        self.pooling_stat = pooling_stat  # store pooling statistic

        self.C = in_channels  # store number of input channels
        self.H, self.W = (  # compute height and width after patch division
            self.image_size[0] // patch_size[0],  # compute height after division by patch height
            self.image_size[1] // patch_size[1],  # compute width after division by patch width
        )  # end computation of height and width
        self.num_patches = self.H * self.W  # compute total number of patches
        self.proj_conv = nn.Conv2d(  # define first convolution layer for projection
            in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False  # set parameters for conv layer
        )  # end first conv layer initialization
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)  # define batch norm for first projection conv
        if spike_mode == "lif":  # if spike mode is 'lif'
            
            # 05/09 (wyjung): Ablation Study #1
            self.proj_lif = LIFNode(step_mode='m', tau=2.0, detach_reset=True, backend="cupy")  # initialize multistep lif node
            # self.proj_lif = nn.Identity()  # replace lif node with identity function for ablation study
            
        elif spike_mode == "plif":  # if spike mode is 'plif'
            self.proj_lif = ParametricLIFNode(  # initialize multistep parametric lif node
            step_mode='m', init_tau=2.0, detach_reset=True, backend="cupy"  # set initial tau and parameters
            )  # end parametric lif node initialization
        self.maxpool = nn.MaxPool2d(  # define first maxpool2d layer
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False  # set parameters for maxpool2d
        )  # end maxpool2d initialization

        self.proj_conv1 = nn.Conv2d(  # define second convolution layer for projection
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )  # end second conv layer initialization
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)  # define batch norm for second conv layer
        if spike_mode == "lif":  # if spike mode is 'lif'
            self.proj_lif1 = LIFNode(  # initialize second multistep lif node
                step_mode='m', tau=2.0, detach_reset=True, backend="cupy"  # set tau and parameters
            )  # end lif node initialization
            
            # replace with nn.Identity() for ablation study
        elif spike_mode == "plif":  # if spike mode is 'plif'
            self.proj_lif1 = ParametricLIFNode(  # initialize second parametric lif node
                step_mode='m', init_tau=2.0, detach_reset=True, backend="cupy"  # set initial tau and parameters
            )  # end parametric lif node initialization
        self.maxpool1 = nn.MaxPool2d(  # define second maxpool2d layer
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False  # set parameters for maxpool2d
        )  # end maxpool2d initialization

        self.proj_conv2 = nn.Conv2d(  # define third convolution layer for projection
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )  # end third conv layer initialization
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)  # define batch norm for third conv layer
        if spike_mode == "lif":  # if spike mode is 'lif'
            self.proj_lif2 = LIFNode(  # initialize third multistep lif node
                step_mode='m', tau=2.0, detach_reset=True, backend="cupy"  # set tau and parameters
            )  # end lif node initialization
        elif spike_mode == "plif":  # if spike mode is 'plif'
            self.proj_lif2 = ParametricLIFNode(  # initialize third parametric lif node
                init_step_mode='m', tau=2.0, detach_reset=True, backend="cupy"  # set initial tau and parameters
            )  # end parametric lif node initialization
        self.maxpool2 = nn.MaxPool2d(  # define third maxpool2d layer
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False  # set parameters for maxpool2d
        )  # end maxpool2d initialization

        self.proj_conv3 = nn.Conv2d(  # define fourth convolution layer for projection
            embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False  # set parameters for conv layer
        )  # end fourth conv layer initialization
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)  # define batch norm for fourth conv layer
        if spike_mode == "lif":  # if spike mode is 'lif'
            self.proj_lif3 = LIFNode(  # initialize fourth multistep lif node
                step_mode='m', tau=2.0, detach_reset=True, backend="cupy"  # set tau and parameters
            )  # end lif node initialization
        elif spike_mode == "plif":  # if spike mode is 'plif'
            self.proj_lif3 = ParametricLIFNode(  # initialize fourth parametric lif node
                init_step_mode='m', tau=2.0, detach_reset=True, backend="cupy"  # set initial tau and parameters
            )  # end parametric lif node initialization
        self.maxpool3 = nn.MaxPool2d(  # define fourth maxpool2d layer
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False  # set parameters for maxpool2d
        )  # end maxpool2d initialization

        self.rpe_conv = nn.Conv2d(  # define convolution layer for rpe
            embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False  # set parameters for rpe conv layer
        )  # end rpe conv layer initialization
        self.rpe_bn = nn.BatchNorm2d(embed_dims)  # define batch norm for rpe conv layer
        if spike_mode == "lif":  # if spike mode is 'lif'
            self.rpe_lif = LIFNode(  # initialize rpe lif node
                step_mode='m', tau=2.0, detach_reset=True, backend="cupy"  # set tau and parameters
            )  # end lif node initialization
        elif spike_mode == "plif":  # if spike mode is 'plif'
            self.rpe_lif = ParametricLIFNode(  # initialize rpe parametric lif node
                init_step_mode='m', init_tau=2.0, detach_reset=True, backend="cupy"  # set initial tau and parameters
            )  # end parametric lif node initialization

    def forward(self, x, hook=None):  # define forward method with optional hook
        T, B, _, H, W = x.shape  # unpack input shape into time, batch, channels, height, and width
        ratio = 1  # initialize ratio for spatial dimension scaling
        x = self.proj_conv(x.flatten(0, 1))  # flatten time and batch dims, then apply first conv layer
        x = self.proj_bn(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()  # apply bn and reshape output
        x = self.proj_lif(x)  # apply lif activation for first projection
        if hook is not None:  # if hook is provided
            hook[self._get_name() + "_lif"] = x.detach()  # store detached output in hook
        x = x.flatten(0, 1).contiguous()  # flatten time and batch dims again
        if self.pooling_stat[0] == "1":  # if first pooling flag is set to '1'
            x = self.maxpool(x)  # apply first maxpool2d layer
            ratio *= 2  # update ratio by multiplying by 2

        x = self.proj_conv1(x)  # apply second conv layer
        x = self.proj_bn1(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()  # apply bn and reshape output
        x = self.proj_lif1(x)  # apply lif activation for second projection
        if hook is not None:  # if hook is provided
            hook[self._get_name() + "_lif1"] = x.detach()  # store detached output in hook
        x = x.flatten(0, 1).contiguous()  # flatten time and batch dims again
        if self.pooling_stat[1] == "1":  # if second pooling flag is set to '1'
            x = self.maxpool1(x)  # apply second maxpool2d layer
            ratio *= 2  # update ratio by multiplying by 2

        x = self.proj_conv2(x)  # apply third conv layer
        x = self.proj_bn2(x).reshape(T, B, -1, H // ratio, W // ratio).contiguous()  # apply bn and reshape output
        x = self.proj_lif2(x)  # apply lif activation for third projection
        if hook is not None:  # if hook is provided
            hook[self._get_name() + "_lif2"] = x.detach()  # store detached output in hook
        x = x.flatten(0, 1).contiguous()  # flatten time and batch dims again
        if self.pooling_stat[2] == "1":  # if third pooling flag is set to '1'
            x = self.maxpool2(x)  # apply third maxpool2d layer
            ratio *= 2  # update ratio by multiplying by 2

        x = self.proj_conv3(x)  # apply fourth conv layer
        x = self.proj_bn3(x)  # apply bn for fourth conv layer
        if self.pooling_stat[3] == "1":  # if fourth pooling flag is set to '1'
            x = self.maxpool3(x)  # apply fourth maxpool2d layer
            ratio *= 2  # update ratio by multiplying by 2

        # residual connection (Membrane Shortcut)
        x_feat = x  # store feature output before lif activation as x_feat
        x = self.proj_lif3(x.reshape(T, B, -1, H // ratio, W // ratio).contiguous())  # reshape x and apply lif activation for fourth projection
        if hook is not None:  # if hook is provided
            hook[self._get_name() + "_lif3"] = x.detach()  # store detached output in hook
        x = x.flatten(0, 1).contiguous()  # flatten time and batch dims again
        x = self.rpe_conv(x)  # apply rpe convolution layer
        x = self.rpe_bn(x)  # apply bn for rpe layer
        x = (x + x_feat).reshape(T, B, -1, H // ratio, W // ratio).contiguous()  # add rpe output with x_feat and reshape

        H, W = H // self.patch_size[0], W // self.patch_size[1]  # compute new height and width based on patch size
        return x, (H, W), hook  # return output, new spatial dimensions, and hook
