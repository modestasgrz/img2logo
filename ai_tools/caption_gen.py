import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

from utils.config import Config

class CaptionGen():

    def __init__(self, config: Config):

        if torch.cuda.is_available():
            self._device = "cuda:0"
            self._torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self._device = "mps"
            self._torch_dtype = torch.float32
        else:
            self._device = "cpu"
            self._torch_dtype = torch.float32


        self._model = AutoModelForCausalLM.from_pretrained(
            config["pretrained_model_name_or_path"], 
            torch_dtype=self._torch_dtype, 
            trust_remote_code=True
        ).to(self._device)

        self._processor = AutoProcessor.from_pretrained(
            config["pretrained_model_name_or_path"], 
            trust_remote_code=True
        )

        self._task_prompt = config["florence2_task_prompt"]


    def inference(
        self, 
        img: Image.Image,
        text_input: str = None
    ) -> str:

        if text_input is None:
            prompt = self._task_prompt
        else:
            prompt = self._task_prompt + text_input

        inputs = self._processor(
            text=prompt, images=img, return_tensors="pt"
        ).to(self._device, self._torch_dtype)

        generated_ids = self._model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self._processor.post_process_generation(
            generated_text, task=self._task_prompt, image_size=(img.width, img.height)
        )[self._task_prompt]

        return parsed_answer