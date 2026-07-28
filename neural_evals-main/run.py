import os
import argparse
from tqdm import *

parser = argparse.ArgumentParser("Training Neural PDE Solvers")

## training
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--epochs", type=int, default=500, help="maximum epochs")
parser.add_argument(
    "--weight_decay", type=float, default=1e-5, help="optimizer weight decay"
)
parser.add_argument("--pct_start", type=float, default=0.3, help="oncycle lr schedule")
parser.add_argument("--batch-size", type=int, default=8, help="batch size")
parser.add_argument("--gpu", type=str, default="0", help="GPU index to use")
parser.add_argument(
    "--max_grad_norm", type=float, default=None, help="make the training stable"
)
parser.add_argument(
    "--use_ema_clip",
    type=int,
    default=0,
    help="Use adaptive EMA-based gradient clipping instead of fixed max_grad_norm",
)
parser.add_argument(
    "--ema_coef1",
    type=float,
    default=0.9,
    help="Fast EMA coefficient for ema grad clip",
)
parser.add_argument(
    "--ema_coef2",
    type=float,
    default=0.99,
    help="Slow EMA coefficient for ema grad clip",
)
parser.add_argument(
    "--ema_max_norm_ratio",
    type=float,
    default=2.0,
    help="Clip trigger: clip when grad_norm > ema_max_norm_ratio * slow_ema",
)
parser.add_argument(
    "--ema_clip_norm_ratio",
    type=float,
    default=1.1,
    help="Clip target: clip grad_norm down to ema_clip_norm_ratio * fast_ema",
)
parser.add_argument(
    "--derivloss",
    type=bool,
    default=False,
    help="adopt the spatial derivate as regularization",
)
parser.add_argument(
    "--teacher_forcing",
    type=int,
    default=1,
    help="adopt teacher forcing in autoregressive to speed up convergence",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="AdamW",
    help="optimizer type, select from Adam, AdamW, Lion",
)
parser.add_argument(
    "--amp",
    type=int,
    default=0,
    help="Enable torch.cuda.amp automatic mixed precision training",
)
parser.add_argument(
    "--scheduler",
    type=str,
    default="OneCycleLR",
    help="learning rate scheduler, select from [OneCycleLR, CosineAnnealingLR, StepLR, WarmupCosine]",
)
parser.add_argument(
    "--step_size", type=int, default=100, help="step size for StepLR scheduler"
)
parser.add_argument(
    "--gamma", type=float, default=0.5, help="decay parameter for StepLR scheduler"
)
parser.add_argument(
    "--min_lr",
    type=float,
    default=0.0,
    help="Minimum learning rate for WarmupCosine scheduler",
)
parser.add_argument(
    "--warmup_fraction",
    type=float,
    default=0.0,
    help="Fraction of total steps used for linear warmup in WarmupCosine scheduler",
)
parser.add_argument(
    "--lion_beta1", type=float, default=0.9, help="Beta1 parameter for Lion optimizer"
)
parser.add_argument(
    "--lion_beta2", type=float, default=0.99, help="Beta2 parameter for Lion optimizer"
)

## data
parser.add_argument("--data_path", type=str, default="/data/fno/", help="data folder")
parser.add_argument("--loader", type=str, default="airfoil", help="type of data loader")
parser.add_argument(
    "--train_ratio", type=float, default=0.8, help="training data ratio"
)
parser.add_argument("--ntrain", type=int, default=1000, help="training data numbers")
parser.add_argument("--ntest", type=int, default=200, help="test data numbers")
parser.add_argument(
    "--normalize", type=bool, default=False, help="make normalization to output"
)
parser.add_argument(
    "--norm_type",
    type=str,
    default="UnitTransformer",
    help="dataset normalize type. select from [UnitTransformer, UnitGaussianNormalizer]",
)
parser.add_argument(
    "--geotype",
    type=str,
    default="unstructured",
    help="select from [unstructured, structured_1D, structured_2D, structured_3D]",
)
parser.add_argument(
    "--time_input", type=bool, default=False, help="for conditional dynamic task"
)
parser.add_argument(
    "--space_dim", type=int, default=2, help="position information dimension"
)
parser.add_argument(
    "--fun_dim", type=int, default=0, help="input observation dimension"
)
parser.add_argument(
    "--out_dim", type=int, default=1, help="output observation dimension"
)
parser.add_argument(
    "--shapelist", type=list, default=None, help="for structured geometry"
)
parser.add_argument(
    "--downsamplex", type=int, default=1, help="downsample rate in x-axis"
)
parser.add_argument(
    "--downsampley", type=int, default=1, help="downsample rate in y-axis"
)
parser.add_argument(
    "--downsamplez", type=int, default=1, help="downsample rate in z-axis"
)
parser.add_argument("--radius", type=float, default=0.2, help="for construct geometry")

## task
parser.add_argument(
    "--task",
    type=str,
    default="steady",
    help="select from [steady, dynamic_autoregressive, dynamic_conditional]",
)
parser.add_argument("--T_in", type=int, default=10, help="for input sequence")
parser.add_argument("--T_out", type=int, default=10, help="for output sequence")

## models
parser.add_argument("--model", type=str, default="Transolver")
parser.add_argument("--n_hidden", type=int, default=64, help="hidden dim")
parser.add_argument("--n_layers", type=int, default=3, help="layers")
parser.add_argument("--n_heads", type=int, default=4, help="number of heads")
parser.add_argument("--act", type=str, default="gelu")
parser.add_argument(
    "--mlp_ratio", type=int, default=1, help="mlp ratio for feedforward layers"
)
parser.add_argument("--dropout", type=float, default=0.0, help="dropout")
parser.add_argument(
    "--unified_pos", type=int, default=0, help="for unified position embedding"
)
parser.add_argument(
    "--ref",
    type=int,
    default=8,
    help="number of reference points for unified pos embedding",
)

## model specific configuration
parser.add_argument(
    "--slice_num", type=int, default=32, help="number of physical states for Transolver"
)
parser.add_argument(
    "--modes", type=int, default=12, help="number of basis functions for LSM and FNO"
)
parser.add_argument("--psi_dim", type=int, default=8, help="number of psi_dim for ONO")
parser.add_argument(
    "--attn_type",
    type=str,
    default="nystrom",
    help="attn_type for ONO, select from nystrom, linear, selfAttention",
)
parser.add_argument(
    "--mwt_k", type=int, default=3, help="number of wavelet basis functions for MWT"
)
parser.add_argument(
    "--mspt_V", type=int, default=32, help="number of chunks for MSPT attention"
)
parser.add_argument(
    "--mspt_Q", type=int, default=1, help="number of pooled tokens per chunk for MSPT"
)
parser.add_argument(
    "--mspt_pool",
    type=str,
    default="mean",
    choices=["mean", "max", "linear"],
    help="pooling strategy for MSPT global tokens",
)
parser.add_argument(
    "--mspt_chunking",
    type=str,
    default="linear",
    choices=["linear", "balltree"],
    help="MSPT point chunking strategy",
)
parser.add_argument(
    "--mspt_use_rope",
    type=int,
    default=0,
    help="Enable rotary positional embeddings inside MSPT attention",
)
parser.add_argument(
    "--mspt_rope_base",
    type=float,
    default=10000.0,
    help="Base for MSPT rotary embedding frequencies",
)
parser.add_argument(
    "--mspt_use_flash_attn",
    type=int,
    default=0,
    help="Enable FlashAttention kernels inside MSPT blocks",
)
parser.add_argument(
    "--mspt_distribute_blocks",
    type=int,
    default=0,
    help="Distribute MSPT blocks across multiple GPUs (experimental)",
)
parser.add_argument(
    "--erwin_c_hidden",
    type=str,
    default="64,128,256,512,1024",
    help="Comma-separated hidden dimensions (encoder + bottleneck) for Erwin",
)
parser.add_argument(
    "--erwin_ball_sizes",
    type=str,
    default="64,64,64,64,64",
    help="Comma-separated ball sizes per Erwin encoder layer (including bottleneck)",
)
parser.add_argument(
    "--erwin_strides",
    type=str,
    default="2,2,2,1",
    help="Comma-separated pooling strides between Erwin encoder layers",
)
parser.add_argument(
    "--erwin_enc_depths",
    type=str,
    default="2,2,6,2,2",
    help="Comma-separated numbers of Erwin blocks per encoder (last entry is bottleneck depth)",
)
parser.add_argument(
    "--erwin_dec_depths",
    type=str,
    default="2,2,2,2",
    help="Comma-separated numbers of Erwin blocks per decoder layer",
)
parser.add_argument(
    "--erwin_enc_heads",
    type=str,
    default="2,4,8,16,16",
    help="Comma-separated list of attention heads per Erwin encoder layer (defaults to n_heads)",
)
parser.add_argument(
    "--erwin_dec_heads",
    type=str,
    default="2,4,8,8",
    help="Comma-separated list of attention heads per Erwin decoder layer (defaults to n_heads)",
)
parser.add_argument(
    "--erwin_rotate",
    type=float,
    default=0.0,
    help="Rotation angle (degrees) used when building Erwin ball trees (0 disables rotations)",
)
parser.add_argument(
    "--erwin_decode",
    type=int,
    default=1,
    help="Whether to run the Erwin decoder (1) or return latent features only (0)",
)
parser.add_argument(
    "--erwin_mlp_ratio",
    type=int,
    default=4,
    help="SwiGLU expansion for Erwin blocks (overrides global mlp_ratio if provided)",
)
parser.add_argument(
    "--erwin_mp_steps",
    type=int,
    default=3,
    help="Number of MPNN steps used in Erwin embedding",
)
parser.add_argument(
    "--erwin_concat_pos",
    type=int,
    default=1,
    help="Concatenate xyz coordinates to node features before feeding Erwin (1=yes, 0=no)",
)
# PDE Transformer specific
parser.add_argument(
    "--pdet_patch_size",
    type=int,
    default=4,
    help="Patch size for PDE_Transformer_SPRINT",
)
parser.add_argument(
    "--pdet_window_size",
    type=int,
    default=8,
    help="Window size for PDE_Transformer_SPRINT",
)
parser.add_argument(
    "--pdet_depth",
    type=str,
    default="2,4,2",
    help="Comma-separated block depths (encoder + latent + decoder) for PDE_Transformer_SPRINT",
)
parser.add_argument(
    "--pdet_max_hidden",
    type=int,
    default=512,
    help="Max hidden dim for PDE_Transformer_SPRINT",
)
parser.add_argument(
    "--pdet_periodic",
    type=int,
    default=0,
    help="Use periodic padding in PDE_Transformer_SPRINT (0/1)",
)
parser.add_argument(
    "--pdet_sprint_drop_mode",
    type=str,
    default="random",
)
parser.add_argument(
    "--pdet_sprint_fusion_type",
    type=str,
    default="linear",
)
parser.add_argument(
    "--pdet_output_act",
    type=str,
    default=None,
)
parser.add_argument(
    "--pdet_use_upsample_act",
    action="store_true",
)
parser.add_argument(
    "--pdet_use_gated_mlp",
    action="store_true",
)
parser.add_argument(
    "--pdet_carrier_tokens",
    action="store_true",
    help="Enable carrier tokens (FasterViT-style cross-window communication) in PDE_Transformer_SPRINT",
)
parser.add_argument(
    "--pdet_pretrained_in_channels",
    type=int,
    default=None,
    help="Optional internal in_channels for PDE_Transformer_SPRINT when adapting from a pretrained backbone",
)
parser.add_argument(
    "--pdet_pretrained_out_channels",
    type=int,
    default=None,
    help="Optional internal out_channels for PDE_Transformer_SPRINT when adapting from a pretrained backbone",
)

parser.add_argument(
    "--pretrained_ckpt",
    type=str,
    default=None,
    help="Optional checkpoint path to initialize model weights before training",
)
parser.add_argument(
    "--pretrained_strict",
    type=int,
    default=0,
    help="Load pretrained checkpoint with strict key/shape matching (1) or shape-filtered partial loading (0)",
)

parser.add_argument(
    "--torch_compile",
    type=int,
    default=0,
    help="Enable torch.compile for the selected model",
)
parser.add_argument(
    "--torch_compile_mode",
    type=str,
    default="default",
    help="torch.compile mode to use when enabled",
)

## eval
parser.add_argument("--eval", type=int, default=0, help="evaluation or not")
parser.add_argument(
    "--save_name", type=str, default="Transolver_check", help="name of folders"
)
parser.add_argument(
    "--vis_num", type=int, default=10, help="number of visualization cases"
)
parser.add_argument(
    "--vis_bound",
    type=int,
    nargs="+",
    default=None,
    help="size of region for visualization, in list",
)
parser.add_argument(
    "--vis_cbar_min",
    type=float,
    default=None,
    help="Fix the minimum value for visualization color bars",
)
parser.add_argument(
    "--vis_cbar_max",
    type=float,
    default=None,
    help="Fix the maximum value for visualization color bars",
)
parser.add_argument(
    "--export_surface_ply",
    type=str,
    default=None,
    help="Directory or file path to export car-design surface predictions as PLY (steady_design only)",
)
parser.add_argument(
    "--export_surface_limit",
    type=int,
    default=1,
    help="Maximum number of validation samples to export as surface PLYs",
)
parser.add_argument(
    "--export_surface_sample",
    type=str,
    default=None,
    help="Specific validation sample identifier (e.g. param0/xxxx) to export; overrides limit filtering",
)
parser.add_argument(
    "--export_surface_include_error",
    type=int,
    default=1,
    help="Whether to include a prediction-error column when exporting surface PLYs",
)

args = parser.parse_args()
eval = args.eval
save_name = args.save_name
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def main():
    if args.task == "steady":
        from exp.exp_steady import Exp_Steady

        exp = Exp_Steady(args)
    elif args.task == "steady_design":
        from exp.exp_steady_design import Exp_Steady_Design

        exp = Exp_Steady_Design(args)
    elif args.task == "dynamic_autoregressive":
        from exp.exp_dynamic_autoregressive import Exp_Dynamic_Autoregressive

        exp = Exp_Dynamic_Autoregressive(args)
    elif args.task == "dynamic_conditional":
        from exp.exp_dynamic_conditional import Exp_Dynamic_Conditional

        exp = Exp_Dynamic_Conditional(args)
    else:
        raise NotImplementedError

    if eval:
        exp.test()
    else:
        exp.train()
        exp.test()


if __name__ == "__main__":
    main()
