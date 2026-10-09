import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_

from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op

from .Coformer import UNetDecoder as Decoder



class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or \
                isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


class TriadPlainConvUNetEncoder(nn.Module):
    """
    Triad PlainConvUNet encoder.
    This matches Triad-PlainConvUNet-MAE.pth.
    forward(x) returns skips.
    """

    def __init__(self, num_input_channels=1):
        super().__init__()

        UNet_base_num_features = 32
        unet_max_num_features = 320

        conv_kernel_sizes = [[3, 3, 3]] * 6
        pool_op_kernel_sizes = [
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ]

        n_conv_per_stage_encoder = [2, 2, 2, 2, 2, 2]

        dim = len(conv_kernel_sizes[0])
        conv_op = convert_dim_to_conv_op(dim)
        num_stages = len(conv_kernel_sizes)

        features_per_stage = [
            min(UNet_base_num_features * 2 ** i, unet_max_num_features)
            for i in range(num_stages)
        ]

        self.encoder = PlainConvEncoder(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=conv_kernel_sizes,
            strides=pool_op_kernel_sizes,
            n_conv_per_stage=n_conv_per_stage_encoder,
            conv_bias=True,
            norm_op=get_matching_instancenorm(conv_op),
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            return_skips=True,
            nonlin_first=False,
        )

    def forward(self, x):
        return self.encoder(x)


class Triad_LWCDecoder_Net(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=None,
        conv_kernel_size=None,
        pool_op_kernel_sizes=None,
        base_num_features=32,
        max_num_features=320,
        pretrained_path=None,
    ):
        super().__init__()

        self.MODEL_NUM_CLASSES = num_classes

        if conv_kernel_size is None:
            conv_kernel_size = [[3, 3, 3]] * 6

        if pool_op_kernel_sizes is None:
            pool_op_kernel_sizes = [
                [1, 1, 1],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
            ]

        embed_dims = [
            min(base_num_features * 2 ** i, max_num_features)
            for i in range(len(conv_kernel_size))
        ]

        stride = pool_op_kernel_sizes
        padding = [[1 if i == 3 else 0 for i in krnl] for krnl in conv_kernel_size]

        # 1. Triad pretrained encoder backbone
        self.backbone = TriadPlainConvUNetEncoder(num_input_channels=in_channels)
        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            self.backbone.load_state_dict(ckpt, strict=True)
        if pretrained_path is not None:
            print(f"[Triad] Loaded pretrained encoder from: {pretrained_path}")

        # 2. Decoder
        self.decoder = Decoder(
            num_class=num_classes,
            embed_dims=embed_dims[::-1],
            kernel_size=conv_kernel_size[:0:-1],
            stride=stride[:0:-1],
            padding=padding[:0:-1],
        )

        # 只初始化 decoder，千万不要 self.apply(...)
        # 否则会把 Triad backbone 预训练权重洗掉
        self.decoder.apply(InitWeights_He())

        backbone_params = sum([p.nelement() for p in self.backbone.parameters()])
        total_params = sum([p.nelement() for p in self.parameters()])

        print("  + Number of Triad Backbone Params: %.2f(e6) M" % (backbone_params / 1e6))
        print("  + Number of Total Params: %.2f(e6) M" % (total_params / 1e6))

    def forward(self, inputs):
        x = inputs
        x = self.backbone(x)
        x = self.decoder(x)
        return x


class Triad_LWCDecoder_UNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        num_classes=36,
        conv_kernel_size=None,
        pool_op_kernel_sizes=None,
        base_num_features=32,
        max_num_features=320,
        deep_supervision=True,
        pretrained_path=None,
    ):
        super().__init__()

        self.deep_supervision = deep_supervision

        self.network = Triad_LWCDecoder_Net(
            in_channels=in_channels,
            num_classes=num_classes,
            conv_kernel_size=conv_kernel_size,
            pool_op_kernel_sizes=pool_op_kernel_sizes,
            base_num_features=base_num_features,
            max_num_features=max_num_features,
            pretrained_path=pretrained_path,
        )

    def forward(self, x):
        seg_output = self.network(x)

        if self.deep_supervision:
            if not isinstance(seg_output, list) and not isinstance(seg_output, tuple):
                return [seg_output]
            else:
                return seg_output
        else:
            if not isinstance(seg_output, list) and not isinstance(seg_output, tuple):
                return seg_output
            else:
                return seg_output[0]
            

if __name__ == '__main__':
    import torch
    model = Triad_LWCDecoder_UNet(
        in_channels=1,
        num_classes=36,
        conv_kernel_size=[[3, 3, 3]] * 6,
        pool_op_kernel_sizes=[
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ],
        deep_supervision=False,
        pretrained_path=None,
    )
    x = torch.rand(1,1,128,128,128)
    y = model(x)
    print(y.shape)
