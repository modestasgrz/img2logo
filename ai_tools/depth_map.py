import torch
import numpy as np

from PIL import Image

from ai_tools.depth_anything_v2.dpt import DepthAnythingV2

class DepthMap():

    def __init__(self, config):

        if torch.cuda.is_available():
            self._device = "cuda:0"
            self._torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self._device = "mps"
            self._torch_dtype = torch.float16
        else:
            self._device = "cpu"
            self._torch_dtype = torch.float32

        self._model = DepthAnythingV2(
            encoder=config['depth_estimator_encoder'], features=64, out_channels=[48, 96, 192, 384]
        )
        self._model.load_state_dict(
            torch.load(config['depth_estimator_name_or_path'], map_location=self._device)
        )
        self._model = self._model.to(self._device)
        
        self._model.eval()


    def inference(self, img: Image.Image) -> Image.Image:
        
        depth = self._model.infer_image(
            np.array(img)
        )

        return Image.fromarray(depth)