import torch

def move_to_device(model):
    if torch.cuda.is_available():
        return model.to("cuda").to(torch.bfloat16)
    elif torch.backends.mps.is_available():
        return model.to("mps")
    return model.to(torch.bfloat16)