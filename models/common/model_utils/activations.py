
import torch

ACT_FN_MAP = {
    "relu": torch.nn.functional.relu,
}


def get_activation_fn(act_str):
    return ACT_FN_MAP[act_str]