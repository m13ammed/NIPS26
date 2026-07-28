from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
from tqdm import tqdm


@dataclass
class VideoPipelineOutput(BaseOutput):
    """
    Output class for video pipelines.

    Args:
        videos (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    videos: Union[List[np.ndarray]]


class VideoPipelineDirect(DiffusionPipeline):
    r"""
    Pipeline for video generation (in a single step). Adapted from DDPMPipeline.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet):
        super().__init__()
        self.register_modules(unet=unet)

    @torch.no_grad()
    def __call__(
        self,
        data: torch.Tensor,
        num_frames: int = 10,
        output_type: Optional[str] = "",
        return_dict: bool = True,
        class_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[VideoPipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            data (`torch.Tensor`):
                The first video frame. The shape should be `(batch_size, num_channels, height, width)`.
            num_frames (`int`, *optional*, defaults to 10):
                The number of frames to generate.
            output_type (`str`, *optional*, defaults to `""`):
                The output format of the generated image. Currently only `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            kwargs (`dict`): Additional keyword arguments to be passed to the model.
        ```

        Returns:
            [`~pipelines.VideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        batch_size = data.shape[0]

        frames = [data.cpu().numpy()]
        previous_frame = data.to(self.device)

        for _ in tqdm(range(num_frames)):
            input = previous_frame
            model_output = self.unet(input, class_labels=class_labels).sample

            previous_frame = model_output
            frames.append(model_output.cpu().numpy())

        vid = np.array(frames)
        vid = np.swapaxes(vid, 0, 1)

        if not return_dict:
            return (vid,)

        return VideoPipelineOutput(videos=vid)
