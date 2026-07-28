from pathlib import Path
from typing import List, Union

from pde.utils import instantiate_from_config
import torch.nn as nn
import lightning
import torch
from tqdm import tqdm
import numpy as np
from data.metadata_remapping import all_pdes, get_long_to_short_pde_map


class SingleStepSupervised(lightning.LightningModule):
    def __init__(
        self,
        model: Union[dict, nn.Module],
        ckpt_path=None,
        ignore_keys=None,
        image_key=0,
        monitor=None,
        detect_zero_grad=False,
        normalize_channels=False,
        optimizer="adamw",
        lr_scheduler="constant",
        warmup_steps=0,
        weight_decay=1e-15,
        accumulate_grad_batches=1,
        muon_learning_rate=1e-3,
        log_norm_every_n_steps=10,
        normalized_loss: bool = False,
        task_balanced_loss: bool = False,
    ):
        super(SingleStepSupervised, self).__init__()

        self.image_key = image_key
        self.optimizer = optimizer
        self.normalize_channels = normalize_channels
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.accumulate_grad_batches = accumulate_grad_batches
        self.batch_size = None  # to be set by trainer
        self.muon_learning_rate = muon_learning_rate
        self.log_norm_every_n_steps = log_norm_every_n_steps
        self.normalized_loss = normalized_loss
        self.task_balanced_loss = task_balanced_loss
        print(f"Normalized Loss Training: {self.normalized_loss}")

        # Muon doesn't support 2D params, therefore needs manual optimization
        if self.optimizer == "muon":
            self.automatic_optimization = False

        if isinstance(model, dict):
            self.model: nn.Module = instantiate_from_config(model)
        else:
            self.model = model

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.downsample_factor = 1
        self.detect_zero_grad = detect_zero_grad

        # Cache dataset names for nicer metric labels
        self._labels_to_name = {i: name for i, name in enumerate(all_pdes)}
        long_to_short = get_long_to_short_pde_map()
        for i, name in enumerate(all_pdes):
            short_name = long_to_short.get(name.lower().replace(":", ""), None)
            if short_name is not None:
                self._labels_to_name[i] = short_name

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, conditioning: torch.Tensor = None
    ) -> torch.Tensor:

        return self.model(x, t)

    def get_pipeline_args(self):
        return {
            "unet": self.model,
        }

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = None):
        if ignore_keys is None:
            ignore_keys = list()

        if Path(path).is_dir():
            path = Path(path).joinpath("last.ckpt")
        else:
            path = Path(path)

        if path.is_file():
            sd = torch.load(path, map_location="cpu")["state_dict"]
            keys = list(sd.keys())
            for k in keys:
                for ik in ignore_keys:
                    if k.startswith(ik):
                        print("Deleting key {} from state_dict.".format(k))
                        del sd[k]
            self.load_state_dict(sd, strict=False)
            print(f"Restored from {path}")

    def get_input(self, batch, batch_dim=True, trim: int = 0):

        data: torch.Tensor = batch["data"]
        meta_data_loading: dict = batch["loading_metadata"]
        meta_data_physical: dict = batch["physical_metadata"]

        if batch_dim:
            x: torch.Tensor = data[:, 0 + trim]
            y: torch.Tensor = data[:, 1 + trim :]

            task_idx = meta_data_physical["PDE"][:, 0]

        else:
            x: torch.Tensor = data[0 + trim]
            y: torch.Tensor = data[1 + trim :]
            task_idx = meta_data_physical["PDE"]

            x = torch.unsqueeze(x, 0)
            y = torch.unsqueeze(y, 0)

            if not torch.is_tensor(task_idx):
                task_idx = torch.tensor(task_idx)

        if self.downsample_factor > 1:
            # downsample with average pooling
            x = nn.functional.avg_pool2d(x, self.downsample_factor)

            num_batches = y.shape[0]
            y = y.reshape(-1, y.shape[-3], y.shape[-2], y.shape[-1])
            y = nn.functional.avg_pool2d(y, self.downsample_factor)
            y = y.reshape(num_batches, -1, y.shape[-3], y.shape[-2], y.shape[-1])

        return x, y, task_idx

    def _per_sample_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        # Mean over channel/spatial dims, keep batch dimension
        spatial_dims = tuple(range(1, pred.dim()))
        mse = ((pred - target) ** 2).mean(dim=spatial_dims)
        if self.normalized_loss:
            target_energy = (target**2).mean(dim=spatial_dims).clamp(min=1e-3)
            mse = mse / target_energy
        return mse

    def _task_balanced_mean(
        self, per_sample_values: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Average within each task, then average across tasks (equal task weight)."""
        if self.task_balanced_loss:
            unique_labels = labels.unique()
            task_means = torch.stack(
                [per_sample_values[labels == t].mean() for t in unique_labels]
            )
            return task_means.mean()
        else:
            return per_sample_values.mean()

    def _equal_task_mean(
        self, per_sample_values: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Always average within each task and then across tasks."""
        unique_labels = labels.unique()
        task_means = torch.stack(
            [per_sample_values[labels == t].mean() for t in unique_labels]
        )
        return task_means.mean()

    def _per_sample_nrmse1(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample nRMSE for one-step prediction, aligned with SimulationMSE update logic."""
        spatial_dims = tuple(range(1, pred.dim()))
        mse = ((pred - target) ** 2).mean(dim=spatial_dims)
        target_energy = (target**2).mean(dim=spatial_dims).clamp(min=1e-3)
        return torch.sqrt(mse / target_energy)

    def _log_per_task_losses(
        self, losses: torch.Tensor, labels: torch.Tensor, prefix: str
    ):
        # Log mean loss per task id present in the batch; prefer dataset names from the datamodule when available.
        # Disabled under DDP: different ranks may have different task keys in their batches, which would cause
        # all_reduce collectives to deadlock (sync_dist=True) or produce inaccurate per-rank-only values (sync_dist=False).
        if self.trainer is not None and self.trainer.world_size > 1:
            return
        # Move labels to CPU to avoid GPU→CPU sync inside .unique()
        labels_cpu = labels.cpu()
        unique_labels = labels_cpu.unique()
        for task_id in unique_labels:
            mask = labels_cpu == task_id
            task_loss = losses[mask.to(losses.device)].mean()

            try:
                name_suffix = (
                    self._labels_to_name[int(task_id)]
                    if self._labels_to_name is not None
                    else str(int(task_id))
                )
            except Exception:
                name_suffix = str(int(task_id))

            self.log(
                f"{prefix}/loss_task_{name_suffix}",
                task_loss,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )

    def _compute_grad_norm(self) -> torch.Tensor:
        # L2 norm across all parameter gradients (skips params without grad)
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if not grads:
            return torch.tensor(0.0, device=self.device)
        return torch.sqrt(sum(g.detach().pow(2).sum() for g in grads))

    def test_step(self, batch, batch_idx):

        return 0, {}

    def training_step(self, batch, batch_idx):
        input, target, labels = self.get_input(batch)
        pred = self.model(input, class_labels=labels).sample
        target = target[:, -1]  # select last frame as target

        if self.normalize_channels:
            # normalize channels (mean, std)
            target = (target - target.mean(dim=(2, 3), keepdim=True)) / (
                target.std(dim=(2, 3), keepdim=True) + 1e-4
            )
            pred = (pred - target.mean(dim=(2, 3), keepdim=True)) / (
                target.std(dim=(2, 3), keepdim=True) + 1e-4
            )

        per_sample_loss = self._per_sample_mse(pred, target)
        loss = self._task_balanced_mean(per_sample_loss, labels)

        importance_reg_loss = 0.0
        inner_model = getattr(self.model, "model", self.model)  # Handle wrapper
        if (
            hasattr(inner_model, "last_importance_logits")
            and inner_model.last_importance_logits is not None
        ):
            logits = inner_model.last_importance_logits  # (B, H, W)
            # Compute entropy over spatial positions - encourage diverse selection
            probs = torch.softmax(logits.view(logits.shape[0], -1), dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
            max_entropy = torch.log(
                torch.tensor(probs.shape[-1], dtype=probs.dtype, device=probs.device)
            )
            importance_reg_loss = 0.01 * (
                max_entropy - entropy
            )  # Penalize deviation from max entropy
            loss = loss + importance_reg_loss

            self.log(
                "train/importance_entropy",
                entropy,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                batch_size=self.batch_size,
            )

        self.log(
            "loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        global_step = self.trainer.global_step if self.trainer is not None else 0
        if global_step % self.log_norm_every_n_steps == 0:
            self.log(
                "nRMSE1",
                self._equal_task_mean(self._per_sample_nrmse1(pred, target), labels),
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size,
            )

        self._log_per_task_losses(per_sample_loss, labels, prefix="train")

        # Track the worst-case sample in the batch
        self.log(
            "train/max_loss",
            per_sample_loss.max(),
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        if self.optimizer == "muon":
            opt_muon, opt_adamw = self.optimizers()
            scaled_loss = loss / self.accumulate_grad_batches

            self.manual_backward(scaled_loss)

            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                opt_muon.step()
                opt_adamw.step()

                opt_muon.zero_grad()
                opt_adamw.zero_grad()

                # Step Schedulers (if using LR schedulers)
                if self.lr_scheduler != "constant":
                    sch_muon, sch_adamw = self.lr_schedulers()
                    sch_muon.step()
                    sch_adamw.step()

            # For manual optimization, training_step should not return loss
            return None

        return loss

    def validation_step(self, batch, batch_idx):

        input, target, labels = self.get_input(batch)

        pred = self.model(input, class_labels=labels).sample

        # select last frame as target
        target = target[:, -1]

        if self.normalize_channels:
            # normalize channels (mean, std)
            target = (target - target.mean(dim=(2, 3), keepdim=True)) / (
                target.std(dim=(2, 3), keepdim=True) + 1e-4
            )
            pred = (pred - pred.mean(dim=(2, 3), keepdim=True)) / (
                pred.std(dim=(2, 3), keepdim=True) + 1e-4
            )

        per_sample_loss = self._per_sample_mse(pred, target)
        loss = self._task_balanced_mean(per_sample_loss, labels)

        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val/nRMSE1",
            self._equal_task_mean(self._per_sample_nrmse1(pred, target), labels),
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        self._log_per_task_losses(per_sample_loss, labels, prefix="val")

        self.log(
            "val/max_loss",
            per_sample_loss.max(),
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        return {"val/loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        # Get optimizers
        if self.optimizer == "muon":
            opt_muon, opt_adamw = self.optimizers()
            muon_lr = opt_muon.param_groups[0]["lr"]
            adam_lr = opt_adamw.param_groups[0]["lr"]

            self.log("train/lr_muon", muon_lr, prog_bar=True, on_step=True)
            self.log("train/lr_adamw", adam_lr, prog_bar=True, on_step=True)
        else:
            # Fallback for standard AdamW
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("train/lr", current_lr, prog_bar=True, on_step=True)

    def on_after_backward(self):
        if self.detect_zero_grad:
            for name, param in self.named_parameters():
                if param.grad is None:
                    print("Detected params without gradient: ", name)

        # Only compute expensive norm metrics every N steps to avoid per-step overhead
        global_step = self.trainer.global_step if self.trainer is not None else 0
        if global_step % self.log_norm_every_n_steps == 0:
            # Log total gradient norm
            grad_norm = self._compute_grad_norm()
            self.log(
                "train/grad_norm",
                grad_norm,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                batch_size=self.batch_size,
            )

            # Log weight norm
            weight_norm = torch.sqrt(
                sum(p.detach().pow(2).sum() for p in self.parameters())
            )
            total_params = sum(p.numel() for p in self.parameters())
            self.log(
                "train/weight_norm",
                weight_norm / total_params,
                prog_bar=False,
                logger=True,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                batch_size=self.batch_size,
            )

    def configure_optimizers(self):
        # Compute total training steps. estimated_stepping_batches accounts for
        # max_steps/max_epochs, number of devices, and gradient accumulation.
        total_steps = self.trainer.estimated_stepping_batches
        print(f"Estimated Training Steps Count: {total_steps}")

        # Helper to create scheduler config
        def get_scheduler(optimizer, total_steps, base_lr):
            if self.lr_scheduler == "constant":
                return None
            elif self.lr_scheduler == "cosine":
                if self.warmup_steps > 0:
                    warmup = torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=0.01,
                        end_factor=1.0,
                        total_iters=self.warmup_steps,
                    )
                    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=total_steps - self.warmup_steps,
                        eta_min=base_lr * 0.01,
                    )
                    scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        schedulers=[warmup, cosine],
                        milestones=[self.warmup_steps],
                    )
                else:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=total_steps, eta_min=base_lr * 0.01
                    )
            elif self.lr_scheduler == "linear":
                if self.warmup_steps > 0:
                    warmup = torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=0.01,
                        end_factor=1.0,
                        total_iters=self.warmup_steps,
                    )
                    linear = torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=1.0,
                        end_factor=0.01,
                        total_iters=total_steps - self.warmup_steps,
                    )
                    scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        schedulers=[warmup, linear],
                        milestones=[self.warmup_steps],
                    )
                else:
                    scheduler = torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=1.0,
                        end_factor=0.01,
                        total_iters=total_steps,
                    )
            elif self.lr_scheduler == "cosine_with_restarts":
                if self.warmup_steps > 0:
                    warmup = torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=0.01,
                        end_factor=1.0,
                        total_iters=self.warmup_steps,
                    )
                    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=5000, T_mult=2, eta_min=base_lr * 0.01
                    )
                    scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        schedulers=[warmup, cosine],
                        milestones=[self.warmup_steps],
                    )
                else:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=5000, T_mult=2, eta_min=base_lr * 0.01
                    )
            else:
                raise ValueError(f"LR scheduler {self.lr_scheduler} not supported")

            return {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "val/loss",
            }

        if self.optimizer == "muon":
            # 1. Filter Parameters
            # Muon gets anything == 2D (weights)
            # AdamW gets anything != 2D (biases, layernorms, embeddings)
            muon_params = [
                p for p in self.parameters() if p.ndim == 2 and p.requires_grad
            ]
            adamw_params = [
                p for p in self.parameters() if p.ndim != 2 and p.requires_grad
            ]

            print(f"Muon params: {len(muon_params)}, AdamW params: {len(adamw_params)}")
            # 2. Instantiate Optimizers
            # Note: Muon doesn't use weight decay in the standard sense
            opt_muon = torch.optim.Muon(muon_params, lr=self.muon_learning_rate)
            opt_adamw = torch.optim.AdamW(
                adamw_params, lr=self.learning_rate, weight_decay=self.weight_decay
            )

            optimizers = [opt_muon, opt_adamw]
            schedulers = []

            # 3. Instantiate Schedulers (One for each optimizer)
            if self.lr_scheduler != "constant":
                sched_muon = get_scheduler(
                    opt_muon, total_steps, self.muon_learning_rate
                )
                sched_adamw = get_scheduler(opt_adamw, total_steps, self.learning_rate)
                # Lightning expects just the scheduler object in manual mode usually,
                # but returning the dict config is safer for consistency
                schedulers = [sched_muon, sched_adamw]

            return optimizers, schedulers

        # --- Standard Handling for non-Muon ---
        elif self.optimizer == "adamw":
            opt = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer == "adabelief":
            from adabelief_pytorch import AdaBelief

            opt = AdaBelief(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")

        scheduler_config = get_scheduler(opt, total_steps, self.learning_rate)
        if scheduler_config:
            return {"optimizer": opt, "lr_scheduler": scheduler_config}
        return [opt]

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def predict_step(
        self, previous_frame, num_inference_steps=100, generator=None, **kwargs
    ):
        # Cast input to model dtype (e.g., bfloat16 when training with mixed precision)
        model_dtype = next(self.model.parameters()).dtype
        previous_frame = previous_frame.to(dtype=model_dtype)

        pred = self.model(previous_frame, **kwargs).sample

        return pred

    def postprocess_output(self, x0, physical_metadata):

        fields = physical_metadata["Fields"].to(x0.device)
        fields = (
            fields > 0
        )  # if field id > 0, then it corresponds to a physical field, padded otherwise
        # set padded values to zero

        if len(fields.shape) == 1:
            fields = fields.unsqueeze(0)

        x0 = torch.einsum("bc..., bc -> bc...", x0, fields)

        return x0

    def predict_raw(
        self,
        input_0,
        input_1,
        labels,
        device,
        num_frames=20,
        generator=None,
    ):

        num_inference_steps = 100

        with torch.no_grad():
            input_0 = input_0.to(device)
            input_1 = input_1.to(device)[:, : num_frames - 1]
            labels = labels.to(device)

            if generator is None:
                generator = torch.Generator(device=input_0.device).manual_seed(2024)

            frames = [input_0.cpu()]
            previous_frame = input_0

            for i in tqdm(range(num_frames - 1)):
                x0 = self.predict_step(
                    previous_frame,
                    generator=generator,
                    class_labels=labels,
                    num_inference_steps=num_inference_steps,
                )

                previous_frame = x0
                frames.append(x0.cpu())

            vid = np.array([frame.numpy() for frame in frames])
            vid = np.swapaxes(vid, 0, 1)

            reference = np.array(
                torch.concat([frames[0].unsqueeze(1), input_1.cpu()], dim=1)
            )

        return vid, reference

    def predict(
        self,
        batch,
        device,
        num_frames=20,
        generator=None,
        output_type="numpy",
        num_inference_steps=100,
        return_dict=True,
        reference_boundary=False,
        batch_dim=True,
        trim: int = 0,
    ):

        boundary_slice = 0

        with torch.no_grad():
            input_0, input_1, labels = self.get_input(
                batch, batch_dim=batch_dim, trim=trim
            )

            input_0 = input_0.to(device)
            input_1 = input_1.to(device)[:, : num_frames - 1]
            labels = labels.to(device)

            if generator is None:
                generator = torch.Generator(device=input_0.device).manual_seed(2024)

            frames = [input_0.cpu()]
            previous_frame = input_0

            for i in tqdm(range(num_frames - 1)):
                x0 = self.predict_step(
                    previous_frame,
                    generator=generator,
                    class_labels=labels,
                    num_inference_steps=num_inference_steps,
                )
                x0 = self.postprocess_output(x0, batch["physical_metadata"])

                # fill boundaries with reference
                if reference_boundary:
                    if len(x0.shape) == 4:  # 2D
                        x0[:, :, 0:boundary_slice, :] = input_1[
                            :, i, :, 0:boundary_slice, :
                        ]
                        x0[:, :, -boundary_slice:, :] = input_1[
                            :, i, :, -boundary_slice:, :
                        ]
                        x0[:, :, :, 0:boundary_slice] = input_1[
                            :, i, :, :, 0:boundary_slice
                        ]
                        x0[:, :, :, -boundary_slice:] = input_1[
                            :, i, :, :, -boundary_slice:
                        ]

                    if len(x0.shape) == 5:  # 3D
                        x0[:, :, 0:boundary_slice, :, :] = input_1[
                            :, i, :, 0:boundary_slice, :, :
                        ]
                        x0[:, :, -boundary_slice:, :, :] = input_1[
                            :, i, :, -boundary_slice:, :, :
                        ]
                        x0[:, :, :, 0:boundary_slice, :] = input_1[
                            :, i, :, :, 0:boundary_slice, :
                        ]
                        x0[:, :, :, -boundary_slice:, :] = input_1[
                            :, i, :, :, -boundary_slice:, :
                        ]
                        x0[:, :, :, :, 0:boundary_slice] = input_1[
                            :, i, :, :, :, 0:boundary_slice
                        ]
                        x0[:, :, :, :, -boundary_slice:] = input_1[
                            :, i, :, :, :, -boundary_slice:
                        ]

                previous_frame = x0
                frames.append(x0.cpu())

            vid = np.array([frame.numpy() for frame in frames])
            vid = np.swapaxes(vid, 0, 1)

            reference = np.array(
                torch.concat([frames[0].unsqueeze(1), input_1.cpu()], dim=1)
            )

            if not batch_dim:
                vid = vid[0]
                reference = reference[0]

        return vid, reference
