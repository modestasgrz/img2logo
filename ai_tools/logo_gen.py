from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import torch
import random

from PIL import Image

from utils.config import Config
from ai_tools.schedulers import get_scheduler
from ai_tools.depth_map import DepthMap

class LogoGen():

    def __init__(self, config: Config):

        if torch.cuda.is_available():
            self._device = "cuda:0"
            self._torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self._device = "mps"
            self._torch_dtype = torch.float16
        else:
            self._device = "cpu"
            self._torch_dtype = torch.float32

        self._config = config
        self._depth_map = DepthMap(config)
        
        controlnet = ControlNetModel.from_pretrained(
            config["controlnet_depth_model_path"],
            use_safetensors=True,
            variant="fp16",
            torch_dtype=torch.float16,
        )

        self._pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            config["pretrained_model_name_or_path"], 
            torch_dtype=self._torch_dtype, 
            use_safetensors=True, 
            controlnet=controlnet,
            variant="fp16"
        )
        self._pipe.scheduler = get_scheduler(self._pipe, name=config["sampler_scheduler"])
        
        self._pipe.load_lora_weights(
           config["logo_lora_name_or_path"],
           adapter_name = "logo_lora_adapter"
        )
        logo_lora_adapter_scales = {
            "text_encoder" : config["logo_lora_clip_weight"],
            "unet" : config["logo_lora_unet_weight"]
        }
        self._pipe.set_adapters("logo_lora_adapter", logo_lora_adapter_scales)

        self._pipe = self._pipe.to(self._device)
        self._pipe.text_encoder = self._pipe.text_encoder.to(self._device)

        # self._pipe.enable_model_cpu_offload()
        print(f"Pipeline device: {self._pipe.device}")
        print(f"ControlNet device: {self._pipe.controlnet.device}")
        print(f"Text encoder device: {self._pipe.text_encoder.device}")


    def _generate_seed(self):
        return random.random() * 10000000000000


    #! Applied for controlled depth map generation
    def inference(
            self, 
            prompt: str, 
            negative_prompt: str,
            seed: int,
            img_depth_map: Image.Image,
            img_height: int,
            img_width: int,
            cfg: float,
            num_inference_steps: int,
            output_type: str = "pil"
        ) -> Image.Image:

        return self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img_depth_map,
            num_inference_steps=num_inference_steps,
            height=img_height,
            width=img_width,
            guidance_scale=cfg,
            output_type=output_type,
            generator=torch.Generator(device = self._device).manual_seed(seed),
            controlnet_conditioning_scale=self._config["controlnet_conditioning_scale"],
            control_guidance_start=self._config["control_guidance_start"],
            control_guidance_end=self._config["control_guidance_end"]
        ).images[0]
    

    def generate_logo(
            self, 
            prompt: str, 
            img: Image.Image,
            minimal: bool = False,
            negative_prompt: str = None,
            seed: int = None,
            img_height: int = None,
            img_width: int = None,
            cfg: float = None,
            num_inference_steps: int = None,
            output_type: str = "pil"
    ):
        
        if negative_prompt is None:
            negative_prompt = self._config["negative_prompt"]

        if seed is None:
            seed = int(self._generate_seed())

        if img_height is None:
            img_height = self._config["img_height"]

        if img_width is None:
            img_width = self._config["img_width"]
        
        if cfg is None:
            cfg = self._config["cfg"]

        if num_inference_steps is None:
            num_inference_steps = self._config["num_inference_steps"]

        if minimal:
            prompt = "b&w, " + prompt
            negative_prompt = "colorful, colorized, " + negative_prompt

        img_depth_map = self._depth_map.inference(img)

        return self.inference(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            img_depth_map=img_depth_map,
            img_height=img_height,
            img_width=img_width,
            cfg=cfg,
            num_inference_steps=num_inference_steps,
            output_type=output_type
        )