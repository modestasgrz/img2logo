from PIL import Image

from utils.config import Config
from ui.ui import launch_interface
from ai_tools.caption_gen import CaptionGen
from ai_tools.logo_gen import LogoGen

import time

CAPTION_GEN_SERVICE = CaptionGen(
    Config("configs/caption_gen_config.json")
)
LOGO_GEN_SERVICE = LogoGen(
    Config("configs/img_gen_config.json")
)

def logo_gen_app(
    minimal: bool,
    img: Image.Image
) -> Image.Image:
    
    start_time = time.time()

    print("Generating prompt")
    img_gen_prompt = CAPTION_GEN_SERVICE.inference(img)
    print("Generated prompt: ", img_gen_prompt)
    # img_gen_prompt = "abstract"
    print("Generating logo: ")
    logo = LOGO_GEN_SERVICE.generate_logo(
        prompt=img_gen_prompt,
        img=img,
        minimal=minimal
    )

    delta_time = time.time() - start_time
    print(f"Time passed: {delta_time:.2f}")
    
    return logo
    

if __name__ == "__main__":

    launch_interface(logo_gen_app=logo_gen_app)