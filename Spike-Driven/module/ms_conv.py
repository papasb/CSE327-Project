# module/ms_conv.py
import torch.nn as nn  # import torch.nn as nn
from timm.models.layers import DropPath  # import droppath from timm.models.layers
# import lif node classes from spikingjelly.activation_based.neuron
from spikingjelly.activation_based.neuron import (
    LIFNode,
    ParametricLIFNode   
)

class Erode(nn.Module):  # define erode class inheriting from nn.module
    def __init__(self) -> None:  # define initializer
        super().__init__()  # call parent initializer
        self.pool = nn.MaxPool3d(  # initialize 3d maxpool layer
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)  # set kernel size, stride, and padding
        )  # end maxpool3d initialization

    def forward(self, x):  # define forward method
        return self.pool(x)  # apply maxpool and return result

class MS_MLP_Conv(nn.Module):  # define ms_mlp_conv class inheriting from nn.module
    def __init__(  # define initializer with parameters
        self,  # self reference
        in_features,  # number of input features
        hidden_features=None,  # number of hidden features, default none
        out_features=None,  # number of output features, default none
        drop=0.0,  # dropout rate, default 0.0
        spike_mode="lif",  # spike mode, default 'lif'
        layer=0,  # layer index, default 0
    ):  # end parameter list
        super().__init__()  # call parent initializer
        out_features = out_features or in_features  # set out_features to in_features if none provided
        hidden_features = hidden_features or in_features  # set hidden_features to in_features if none provided
        self.res = in_features == hidden_features  # set residual flag if in_features equals hidden_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)  # initialize first conv layer
        self.fc1_bn = nn.BatchNorm2d(hidden_features)  # initialize batch norm for first conv
        if spike_mode == "lif":  # if spike mode is 'lif'
            self.fc1_lif = LIFNode(step_mode='m', tau=2.0,detach_reset=True, backend="cupy")  # initialize first lif node
        elif spike_mode == "plif":  # if spike mode is 'plif'
            self.fc1_lif = ParametricLIFNode(  # initialize first parametric lif node
                step_mode='m', init_tau=2.0, detach_reset=True, backend="cupy"  # set initial tau and parameters
            )  # end parametric lif node initialization

        self.fc2_conv = nn.Conv2d(  # initialize second conv layer
            hidden_features, out_features, kernel_size=1, stride=1  # set in/out channels and kernel parameters
        )  # end second conv layer initialization
        self.fc2_bn = nn.BatchNorm2d(out_features)  # initialize batch norm for second conv
        if spike_mode == "lif":  # if spike mode is 'lif'
            self.fc2_lif = LIFNode(step_mode='m', tau=2.0,detach_reset=True, backend="cupy")  # initialize second lif node
        elif spike_mode == "plif":  # if spike mode is 'plif'
            self.fc2_lif = ParametricLIFNode(  # initialize second parametric lif node
                step_mode='m', init_tau=2.0, detach_reset=True, backend="cupy"  # set initial tau and parameters
            )  # end parametric lif node initialization

        self.c_hidden = hidden_features  # store hidden features count
        self.c_output = out_features  # store output features count
        self.layer = layer  # store layer index

    def forward(self, x, hook=None):  # define forward method with optional hook
        T, B, C, H, W = x.shape  # unpack input shape into time, batch, channels, height, width
        identity = x  # store input for residual connection

        x = self.fc1_lif(x)  # apply first lif activation
        if hook is not None:  # if hook is provided
            hook[self._get_name() + str(self.layer) + "_fc1_lif"] = x.detach()  # store detached output in hook
        x = self.fc1_conv(x.flatten(0, 1))  # flatten time and batch dims, apply first conv layer
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()  # apply bn and reshape output
        if self.res:  # if residual connection is enabled
            x = identity + x  # add identity to output
            identity = x  # update identity
        x = self.fc2_lif(x)  # apply second lif activation
        if hook is not None:  # if hook is provided
            hook[self._get_name() + str(self.layer) + "_fc2_lif"] = x.detach()  # store detached output in hook
        x = self.fc2_conv(x.flatten(0, 1))  # flatten dims, apply second conv layer
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()  # apply bn and reshape output

        x = x + identity  # add residual connection
        return x, hook  # return output and hook

class MS_SSA_Conv(nn.Module):  # define ms_ssa_conv class inheriting from nn.module
    def __init__(  # define initializer with parameters
        self,  # self reference
        dim,  # feature dimension
        num_heads=8,  # number of attention heads, default 8
        qkv_bias=False,  # flag for qkv bias, default false
        qk_scale=None,  # scaling factor for qk, default none
        attn_drop=0.0,  # attention dropout rate, default 0.0
        proj_drop=0.0,  # projection dropout rate, default 0.0
        sr_ratio=1,  # spatial reduction ratio, default 1
        mode="direct_xor",  # attention mode, default 'direct_xor'
        spike_mode="lif",  # spike mode, default 'lif'
        dvs=False,  # dvs flag, default false
        layer=0,  # layer index, default 0
    ):  # end parameter list
        super().__init__()  # call parent initializer
        assert (  # ensure dimension is divisible by number of heads
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."  # error message if assertion fails
        self.dim = dim  # store feature dimension
        self.dvs = dvs  # store dvs flag
        self.num_heads = num_heads  # store number of attention heads
        if dvs:  # if dvs mode is enabled
            self.pool = Erode()  # initialize erode pooling module
        self.scale = 0.125  # set scaling factor
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)  # initialize q convolution layer
        self.q_bn = nn.BatchNorm2d(dim)  # initialize batch norm for q
        if spike_mode == "lif":  # if spike mode is 'lif'
            self.q_lif = LIFNode(step_mode='m', tau=2.0,detach_reset=True, backend="cupy")  # initialize q lif node
        elif spike_mode == "plif":  # if spike mode is 'plif'
            self.q_lif = ParametricLIFNode(  # initialize q parametric lif node
                step_mode='m', init_tau=2.0, detach_reset=True, backend="cupy"  # set initial tau and parameters
            )  # end parametric lif node initialization

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)  # initialize k convolution layer
        self.k_bn = nn.BatchNorm2d(dim)  # initialize batch norm for k
        if spike_mode == "lif":  # if spike mode is 'lif'
            self.k_lif = LIFNode(step_mode='m', tau=2.0,detach_reset=True, backend="cupy")  # initialize k lif node
        elif spike_mode == "plif":  # if spike mode is 'plif'
            self.k_lif = ParametricLIFNode(  # initialize k parametric lif node
                step_mode='m', init_tau=2.0, detach_reset=True, backend="cupy"  # set initial tau and parameters
            )  # end parametric lif node initialization

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)  # initialize v convolution layer
        self.v_bn = nn.BatchNorm2d(dim)  # initialize batch norm for v
        if spike_mode == "lif":  # if spike mode is 'lif'
            self.v_lif = LIFNode(step_mode='m', tau=2.0,detach_reset=True, backend="cupy")  # initialize v lif node
        elif spike_mode == "plif":  # if spike mode is 'plif'
            self.v_lif = ParametricLIFNode(  # initialize v parametric lif node
                step_mode='m', init_tau=2.0, detach_reset=True, backend="cupy"  # set initial tau and parameters
            )  # end parametric lif node initialization

        if spike_mode == "lif":  # if spike mode is 'lif'
            self.attn_lif = LIFNode(  # initialize attention lif node
                step_mode='m', tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"  # set tau, threshold, and parameters
            )  # end attention lif node initialization
        elif spike_mode == "plif":  # if spike mode is 'plif'
            self.attn_lif = ParametricLIFNode(  # initialize attention parametric lif node
                step_mode='m', init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"  # set initial tau, threshold, and parameters
            )  # end parametric lif node initialization

        self.talking_heads = nn.Conv1d(  # initialize talking heads convolution layer
            num_heads, num_heads, kernel_size=1, stride=1, bias=False  # set in/out channels and kernel parameters
        )  # end talking heads conv initialization
        if spike_mode == "lif":  # if spike mode is 'lif'
            self.talking_heads_lif = LIFNode(  # initialize talking heads lif node
                step_mode='m', tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"  # set tau, threshold, and parameters
            )  # end talking heads lif node initialization
        elif spike_mode == "plif":  # if spike mode is 'plif'
            self.talking_heads_lif = ParametricLIFNode(  # initialize talking heads parametric lif node
                step_mode='m', init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"  # set initial tau, threshold, and parameters
            )  # end parametric lif node initialization

        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)  # initialize projection convolution layer
        self.proj_bn = nn.BatchNorm2d(dim)  # initialize batch norm for projection

        if spike_mode == "lif":  # if spike mode is 'lif'
            self.shortcut_lif = LIFNode(  # initialize shortcut lif node
                step_mode='m', tau=2.0, detach_reset=True, backend="cupy"  # set tau and parameters
            )  # end shortcut lif node initialization
        elif spike_mode == "plif":  # if spike mode is 'plif'
            self.shortcut_lif = ParametricLIFNode(  # initialize shortcut parametric lif node
                step_mode='m', init_tau=2.0, detach_reset=True, backend="cupy"  # set initial tau and parameters
            )  # end parametric lif node initialization

        self.mode = mode  # store attention mode
        self.layer = layer  # store layer index

    def forward(self, x, hook=None):  # define forward method with optional hook
        T, B, C, H, W = x.shape  # unpack input shape into time, batch, channels, height, width
        identity = x  # store input for residual connection
        N = H * W  # compute total number of spatial positions
        x = self.shortcut_lif(x)  # apply shortcut lif activation
        if hook is not None:  # if hook is provided
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()  # store detached output in hook

        x_for_qkv = x.flatten(0, 1)  # flatten time and batch dims for qkv computation
        q_conv_out = self.q_conv(x_for_qkv)  # apply q convolution
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()  # apply bn and reshape output
        q_conv_out = self.q_lif(q_conv_out)  # apply q lif activation

        if hook is not None:  # if hook is provided
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()  # store detached output in hook
        q = (  # compute q tensor for attention
            q_conv_out.flatten(3)  # flatten spatial dimensions
            .transpose(-1, -2)  # transpose last two dims
            .reshape(T, B, N, self.num_heads, C // self.num_heads)  # reshape into attention format
            .permute(0, 1, 3, 2, 4)  # permute dims to arrange heads
            .contiguous()  # ensure contiguous memory layout
        )  # end q tensor computation

        k_conv_out = self.k_conv(x_for_qkv)  # apply k convolution
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()  # apply bn and reshape output
        k_conv_out = self.k_lif(k_conv_out)  # apply k lif activation
        if self.dvs:  # if dvs mode is enabled
            k_conv_out = self.pool(k_conv_out)  # apply pooling on k output
        if hook is not None:  # if hook is provided
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()  # store detached output in hook
        k = (  # compute k tensor for attention
            k_conv_out.flatten(3)  # flatten spatial dimensions
            .transpose(-1, -2)  # transpose last two dims
            .reshape(T, B, N, self.num_heads, C // self.num_heads)  # reshape into attention format
            .permute(0, 1, 3, 2, 4)  # permute dims to arrange heads
            .contiguous()  # ensure contiguous memory layout
        )  # end k tensor computation

        v_conv_out = self.v_conv(x_for_qkv)  # apply v convolution
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()  # apply bn and reshape output
        v_conv_out = self.v_lif(v_conv_out)  # apply v lif activation
        if self.dvs:  # if dvs mode is enabled
            v_conv_out = self.pool(v_conv_out)  # apply pooling on v output
        if hook is not None:  # if hook is provided
            hook[self._get_name() + str(self.layer) + "_v_lif"] = v_conv_out.detach()  # store detached output in hook
        v = (  # compute v tensor for attention
            v_conv_out.flatten(3)  # flatten spatial dimensions
            .transpose(-1, -2)  # transpose last two dims
            .reshape(T, B, N, self.num_heads, C // self.num_heads)  # reshape into attention format
            .permute(0, 1, 3, 2, 4)  # permute dims to arrange heads
            .contiguous()  # ensure contiguous memory layout
        )  # end v tensor computation; shape: t, b, head, n, c//h

        kv = k.mul(v)  # perform element-wise multiplication of k and v
        if hook is not None:  # if hook is provided
            hook[self._get_name() + str(self.layer) + "_kv_before"] = kv  # store kv before pooling in hook
        if self.dvs:  # if dvs mode is enabled
            kv = self.pool(kv)  # apply pooling on kv tensor
        kv = kv.sum(dim=-2, keepdim=True)  # sum over spatial positions
        kv = self.talking_heads_lif(kv)  # apply talking heads lif activation
        if hook is not None:  # if hook is provided
            hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()  # store detached kv in hook
        x = q.mul(kv)  # element-wise multiply q with processed kv
        if self.dvs:  # if dvs mode is enabled
            x = self.pool(x)  # apply pooling on result
        if hook is not None:  # if hook is provided
            hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()  # store detached output in hook

        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()  # transpose and reshape x back to original dims
        x = (  # apply projection layer
            self.proj_bn(self.proj_conv(x.flatten(0, 1)))  # flatten dims, apply proj conv and bn
            .reshape(T, B, C, H, W)  # reshape back to original dimensions
            .contiguous()  # ensure contiguous memory layout
        )  # end projection layer
        x = x + identity  # add residual connection
        return x, v, hook  # return output, v tensor, and hook

class MS_Block_Conv(nn.Module):  # define ms_block_conv class inheriting from nn.module
    def __init__(  # define initializer with parameters
        self,  # self reference
        dim,  # feature dimension
        num_heads,  # number of attention heads
        mlp_ratio=4.0,  # mlp ratio, default 4.0
        qkv_bias=False,  # flag for qkv bias, default false
        qk_scale=None,  # scaling factor for qk, default none
        drop=0.0,  # dropout rate, default 0.0
        attn_drop=0.0,  # attention dropout rate, default 0.0
        drop_path=0.0,  # drop path rate, default 0.0
        norm_layer=nn.LayerNorm,  # normalization layer, default nn.layernorm
        sr_ratio=1,  # spatial reduction ratio, default 1
        attn_mode="direct_xor",  # attention mode, default 'direct_xor'
        spike_mode="lif",  # spike mode, default 'lif'
        dvs=False,  # dvs flag, default false
        layer=0,  # layer index, default 0
    ):  # end parameter list
        super().__init__()  # call parent initializer
        self.attn = MS_SSA_Conv(  # initialize attention module using ms_ssa_conv
            dim,  # pass feature dimension
            num_heads=num_heads,  # pass number of heads
            qkv_bias=qkv_bias,  # pass qkv bias flag
            qk_scale=qk_scale,  # pass qk scale factor
            attn_drop=attn_drop,  # pass attention dropout rate
            proj_drop=drop,  # pass projection dropout rate
            sr_ratio=sr_ratio,  # pass spatial reduction ratio
            mode=attn_mode,  # pass attention mode
            spike_mode=spike_mode,  # pass spike mode
            dvs=dvs,  # pass dvs flag
            layer=layer,  # pass layer index
        )  # end attention module initialization
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()  # initialize drop path or identity
        mlp_hidden_dim = int(dim * mlp_ratio)  # compute hidden dimension for mlp module
        self.mlp = MS_MLP_Conv(  # initialize mlp module using ms_mlp_conv
            in_features=dim,  # pass input feature dimension
            hidden_features=mlp_hidden_dim,  # pass hidden features dimension
            drop=drop,  # pass dropout rate
            spike_mode=spike_mode,  # pass spike mode
            layer=layer,  # pass layer index
        )  # end mlp module initialization

    def forward(self, x, hook=None):  # define forward method with optional hook
        x_attn, attn, hook = self.attn(x, hook=hook)  # apply attention module and retrieve outputs
        x, hook = self.mlp(x_attn, hook=hook)  # apply mlp module on attention output
        return x, attn, hook  # return final output, attention map, and hook
