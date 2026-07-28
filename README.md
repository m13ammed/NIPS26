# PDE-LTD Transformer

This codebase is adapted from [PDE-Transformer](https://github.com/tum-pbs/pde-transformer).

## Environment Setup

1. Install pytorch 2.9.1, torchvision 0.24.1 and flash attention in your python environment according to your system configuration. Other pytorch versions my work but we haven't tested them.
2. Install required packages from `requirements.txt`.
3. Download the datasets or prepare them using the provided scripts in `data/scripts`, either `generate_all_2d.sh` or `simulation.py`. Keep in mind that the datasets take up to a few days to generate depending on your system, and take ~75 GB for the 256x256 resolution and ~275 GB for the 512x512 resolution. To generate the datasets you will additionally need to install the requirements listed in `data/scripts/requirementsExponax.txt`.
4. Set `2D_APE_xxl` in `configs/local.yaml` to the path of the dataset. There, you can also modify the batch size and number of devices if needed.
5. *Optional:* If you want to log your training runs to Weights and Biases run `wandb login` first.

## Training & Ablations

You can run the training script like this:

```bash
python main.py -c configs/CONFIG.yaml -n NAME_OF_RUN [overrides]
```

Where `CONFIG` is either `pde-mc-mse` for the original model or `pde-ltd-mc-mse` for PDE-LTD. Setting the training options in the yaml files should be straightforward, and any overrides can be passed in the command line. For example, to train PDE-LTD you can run:

```bash
python main.py -c configs/pde-ltd-mc-mse.yaml -n pde-ltd-mc-s-mse \
  model.params.model.params.output_activation=gelu \
  model.params.model.params.use_upsample_activation=True \
  model.params.model.params.sprint_drop_mode=learned \
  model.params.model.params.use_gated_mlp=True \
  model.params.model.params.sprint_fusion_type=gated
```

You should modify the `_file` option under `data` in the configuration if you wish to use the smaller subset of the dataset, which was used for ablations. For that you only need to change `ape_2d_multi_task_norm.yaml` to `ape_2d_some_task_norm.yaml`.

Additionally, if you want to run the ablation studies for PDE-LTD in a single command, you can use the script `run_nrms_ablation.sh` which will run all the ablations sequentially.



### Second Benchmark 

The second benchmakr were done using the original repository code and datasets: [thuml/Neural-Solver-Library: A Library for Advanced Neural PDE Solvers.](https://github.com/thuml/Neural-Solver-Library). Only one L4 GPU with 24GB of memory was used, which might have been limiting for MSPT as it was regularly at 100% utilization. The hyperparameters for MSPT and Transolver are as defined in the aforementioned repository, but for PDE-Transformer we found that not all the improvements we came up with for training on the WELL datasets were helpful, and some options were helpful on some benchmark and not on others.

go inside the neural_evals-main to run the experiments and run use the ``` run.py ``` file along with the config flags

#### Common Configuration
```sh
--pdet_window_size 8 \
--pdet_max_hidden 512 \
--epochs 500 \
--lr 5.5e-04 \
```

#### Plasticity
```sh
--n_hidden 96 \
--n_heads 8 \
--mlp_ratio 4 \
--pdet_patch_size 4 \
--pdet_depth 2,4,2 \
--pdet_sprint_drop_mode l2 \
--pdet_sprint_fusion_type gated \
--pdet_output_act gelu \
--pdet_use_upsample_act \
--pdet_use_gated_mlp \
--pdet_max_hidden 512 \
--batch-size 8 \
```
#### Navier Stokes
```sh
--n_hidden 128 \
--n_heads 8 \
--mlp_ratio 4 \
--pdet_patch_size 1 \
--pdet_depth 2,4,2 \
--pdet_output_act gelu \
--pdet_use_upsample_act \
--pdet_periodic 1 \
--batch-size 2
```
#### Pipe
```sh
--n_hidden 96 \
--n_heads 8 \
--mlp_ratio 3 \
--pdet_patch_size 2 \
--pdet_depth 1,3,1 \
--weight_decay 1e-4 \
--batch-size 4
```
#### Airfoil
Increasing the parameter count or decreasing the patch size causes the training to become unstable on this benchmark.
```sh
--n_hidden 96 \
--n_heads 8 \
--mlp_ratio 3 \
--pdet_patch_size 2 \
--pdet_depth 1,3,1 \
--weight_decay 1e-4 \
--batch-size 8 \
--pdet_carrier_tokens
```
#### Darcy
```sh
--n_hidden 96 \
--n_heads 4 \
--mlp_ratio 4 \
--pdet_patch_size 1 \
--pdet_depth 2,4,2 \
--batch-size 4 \
--pdet_carrier_tokens
