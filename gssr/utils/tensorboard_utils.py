import torch
from torchtyping import TensorType
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Optional, Tuple, Type, Union


def write_image(writer: SummaryWriter, name: str, image: TensorType["H", "W", "C"], step: int) -> None:
    """method to write out image

    Args:
        name: data identifier
        image: rendered image to write
        step: the time step to log
    """
    writer.add_images(name, image[None], global_step=step)


def write_scalar(writer: SummaryWriter, name: str, scalar: Union[float, torch.Tensor], step: int) -> None:
    """Required method to write a single scalar value to the logger

    Args:
        name: data identifier
        scalar: value to write out
        step: the time step to log
    """
    writer.add_scalar(name, scalar.item(), step)


def write_scalar_dict(writer: SummaryWriter, name: str, scalar_dict: Dict[str, Any], step: int) -> None:
    """Function that writes out all scalars from a given dictionary to the logger

    Args:
        scalar_dict: dictionary containing all scalar values with key names and quantities
        step: the time step to log
    """
    for key, scalar in scalar_dict.items():
        write_scalar(writer, name + "/" + key, scalar, step)