# model/spikeformer.py
# import partial function from functools
from functools import partial  
# import pytorch library
import torch  
# import torch.nn module as nn
import torch.nn as nn  
# import torch.nn.functional module as f
import torch.nn.functional as F  
# import trunc_normal_ from timm.models.layers
from timm.models.layers import trunc_normal_
# import register_model from timm.models.registry
from timm.models.registry import register_model  
# import _cfg from timm.models.vision_transformer
from timm.models.vision_transformer import _cfg  
# import lif node classes from spikingjelly.activation_based.neuron
from spikingjelly.activation_based.neuron import (
    LIFNode,
    ParametricLIFNode   
)
# import everything from module
from module import *  


# define spike driven transformer class inheriting from nn.module
class SpikeDrivenTransformer(nn.Module):      
    # define initializer method
    def __init__(  
        self,  # self reference
        img_size_h=128,  # image height default 128
        img_size_w=128,  # image width default 128
        patch_size=16,  # patch size default 16
        in_channels=2,  # number of input channels default 2
        num_classes=11,  # number of classes default 11
        embed_dims=512,  # embedding dimensions default 512
        num_heads=8,  # number of attention heads default 8
        mlp_ratios=4,  # mlp ratio default 4
        
        qkv_bias=False,  # flag for qkv bias default false
        # Since Transformers already use norm layers such as BN/LN, data is centered @ mean=0 and std=1, eliminating the need for bias.
        # For Transformers/CNN layers, adding a bias term adds unnecessary overhead to the model.
        
        qk_scale=None,  # scaling factor for qk, default none
        # Scaled dot-product attention: Multiplying Q, K, T causes outputs to expand relative to dim(Q) & dim(K).
        # This causes either extremely low or high values, leading to vanishing/exploding gradients.
        # d(\subscript)k, i.e., the dimenstionality of Q and K, are added in softmax:
        # Attention(Q, K, V) = softmax((Q K^T V) / d(\subscript)k) * V
        
        drop_rate=0.0,  # dropout rate default 0.0
        # Dropout: A regularization approach that prevents overfitting by nullifying/introducing noise while computing each internal layer.
        
        attn_drop_rate=0.0,  # attention dropout rate default 0.0
        drop_path_rate=0.0,  # drop path rate default 0.0
        norm_layer=nn.LayerNorm,  # normalization layer default nn.layernorm
        depths=[6, 8, 6],  # depths for blocks, default list [6,8,6]
        sr_ratios=[8, 4, 2],  # spatial reduction ratios, default list [8,4,2]
        T=4,  # time steps default 4
        pooling_stat="1111",  # pooling statistic default string '1111'
        
        attn_mode="direct_xor",  # attention mode default 'direct_xor'
        # XOR-based binary attention
        
        spike_mode="lif",  # spike mode default 'lif'
        get_embed=False,  # flag for getting embed default false
        dvs_mode=False,  # dvs mode flag default false
        TET=False,  # tet flag default false
        cml=False,  # cml flag default false
        pretrained=False,  # pretrained flag default false
        pretrained_cfg=None,  # pretrained configuration default none
    ):  # end of initializer parameter list
        
        super().__init__()  # initialize parent class
        self.num_classes = num_classes  # assign number of classes
        self.depths = depths  # assign depths configuration

        self.T = T  # assign time steps
        self.TET = TET  # assign tet flag
        self.dvs = dvs_mode  # assign dvs mode flag

        # create list for drop path rates
        dpr = [  
            # compute drop path rate for each depth
            x.item() for x in torch.linspace(0, drop_path_rate, depths)  
        ]  # stochastic depth decay rule

        # create patch embedding using ms_sps
        patch_embed = MS_SPS(  
            img_size_h=img_size_h,  # pass image height
            img_size_w=img_size_w,  # pass image width
            patch_size=patch_size,  # pass patch size
            in_channels=in_channels,  # pass input channels
            embed_dims=embed_dims,  # pass embedding dimensions
            pooling_stat=pooling_stat,  # pass pooling statistic
            spike_mode=spike_mode,  # pass spike mode
        )  # end patch embed instantiation

        # create module list for blocks
        blocks = nn.ModuleList(  
            [  # start list comprehension for blocks
                MS_Block_Conv(  # create ms_block_conv instance
                    dim=embed_dims,  # pass embedding dimensions to block
                    num_heads=num_heads,  # pass number of heads to block
                    mlp_ratio=mlp_ratios,  # pass mlp ratio to block
                    qkv_bias=qkv_bias,  # pass qkv bias flag to block
                    qk_scale=qk_scale,  # pass qk scale to block
                    drop=drop_rate,  # pass dropout rate to block
                    attn_drop=attn_drop_rate,  # pass attention dropout rate to block
                    drop_path=dpr[j],  # pass drop path rate for current block
                    norm_layer=norm_layer,  # pass normalization layer to block
                    sr_ratio=sr_ratios,  # pass spatial reduction ratio to block
                    attn_mode=attn_mode,  # pass attention mode to block
                    spike_mode=spike_mode,  # pass spike mode to block
                    dvs=dvs_mode,  # pass dvs mode flag to block
                    layer=j,  # pass current layer index
                )  # end ms_block_conv instantiation
                for j in range(depths)  # iterate over range of depths
            ]  # end list comprehension
        )  # end module list instantiation for blocks

        setattr(self, f"patch_embed", patch_embed)  # set patch_embed attribute
        setattr(self, f"block", blocks)  # set block attribute

        # classification head
        if spike_mode in ["lif", "alif", "blif"]:  # if spike mode is lif, alif, or blif
            self.head_lif = LIFNode(step_mode='m', tau=2.0, detach_reset=True, backend="cupy")
            # replaces MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")  # create multi-step lif node with tau 2.0
        elif spike_mode == "plif":  # if spike mode is plif
            self.head_lif = ParametricLIFNode(step_mode='m', init_tau=2.0, detach_reset=True, backend="cupy")
            
        self.head = (  # create classification head
            nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()  # use linear layer if num_classes > 0 else identity
        )  # end classification head instantiation
        self.apply(self._init_weights)  # apply weight initialization
        
    # hooks for tensor extraction & data flow visualization
    def _init_hooks(self):
        self.hook = {}

    # define weights initialization function
    def _init_weights(self, m):  
        if isinstance(m, nn.Conv2d):  # if module is convolutional layer
            trunc_normal_(m.weight, std=0.02)  # apply trunc_normal_ to weights with std 0.02
            if m.bias is not None:  # if convolutional layer has bias
                nn.init.constant_(m.bias, 0)  # initialize bias to 0
        elif isinstance(m, nn.BatchNorm2d):  # if module is batchnorm layer
            nn.init.constant_(m.bias, 0)  # initialize bias to 0
            nn.init.constant_(m.weight, 1.0)  # initialize weight to 1.0

    # define forward_features function
    def forward_features(self, x, hook=None):  
        block = getattr(self, f"block")  # get block attribute
        patch_embed = getattr(self, f"patch_embed")  # get patch_embed attribute

        x, _, hook = patch_embed(x, hook=hook)  # apply patch embedding to x
        for blk in block:  # iterate over blocks
            x, _, hook = blk(x, hook=hook)  # apply each block to x

        x = x.flatten(3).mean(3)  # flatten and average x over dimension 3
        return x, hook  # return processed x and hook

    # define forward function
    def forward(self, x, hook=None):  
        if hook is None:
            hook = {} # initialize hook if empty
        if len(x.shape) < 5:  # if input x has less than 5 dimensions
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)  # add time dimension and repeat x for T steps
        else:  # else if x already has 5 dimensions
            x = x.transpose(0, 1).contiguous()  # transpose x to have time dimension first
            p

        x, hook = self.forward_features(x, hook=hook)  # extract features and hook from x
        x = self.head_lif(x)  # apply lif neuron head to x
        if hook is not None:  # if hook is not none
            hook["head_lif"] = x.detach()  # store detached x in hook with key 'head_lif'

        x = self.head(x)  # apply classification head to x
        if not self.TET:  # if tet flag is false
            x = x.mean(0)  # average x over time dimension if tet is false
        return x, hook  # return final x and hook


# register model decorator
@register_model  
# define sdt model function with kwargs
def sdt(**kwargs):  
    model = SpikeDrivenTransformer(  # instantiate spike driven transformer model
        **kwargs,  # pass keyword arguments to model
    )  # end model instantiation
    model.default_cfg = _cfg()  # assign default configuration to model
    return model  # return model
