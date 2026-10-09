from .network_architecture.Triad import Triad_LWCDecoder_UNet


def build_model(modelname='Triad_UNet', in_channels=1, num_classes=36):
    """Build the paper model.

    GitHub release intentionally keeps only the architecture used in the paper
    training pipeline.
    """
    if modelname == 'Triad_UNet':
        model = Triad_LWCDecoder_UNet(
            in_channels=in_channels,
            num_classes=num_classes,
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
    elif modelname == 'Triad_UNet_raw':
        model = Triad_LWCDecoder_UNet(
            in_channels=in_channels,
            num_classes=num_classes,
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
    else:
        raise ValueError(f"Unsupported modelname for paper release: {modelname}")

    return model

if __name__ == '__main__':
    import torch
    model = build_model('Triad_UNet')
    x = torch.rand(1,1,128,128,128)
    y = model(x)
    print(y.shape)

# plan =  {
#             "batch_size": 2,
#             "patch_size": [128,128,128],
#             "spacing": [1.0,1.0,1.0],
#             "use_mask_for_norm": [False],
#             "UNet_base_num_features": 32,
#             "n_conv_per_stage_encoder": [2,2,2,2,2,2],
#             "n_conv_per_stage_decoder": [2,2,2,2,2],
#             "num_pool_per_axis": [5,5,5],
#             "pool_op_kernel_sizes": [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
#             "conv_kernel_sizes": [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
#             "unet_max_num_features": 320,
#             "resampling_fn_data": "resample_data_or_seg_to_shape",
#             "resampling_fn_seg": "resample_data_or_seg_to_shape",
#             "batch_dice": False
#         }
