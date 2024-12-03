from diffusers import EulerAncestralDiscreteScheduler, DiffusionPipeline

def get_scheduler(
        pipe: DiffusionPipeline,
        name: str = "euler a"
    ):

    if name == "euler a":
        return EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        raise NotImplementedError("Specified scheduler not implemented.")