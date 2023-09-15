import torch

def move_to_device(model):
    if torch.cuda.is_available():
        return model.to("cuda")
    elif torch.backends.mps.is_available():
        return model.to("mps")
    return model