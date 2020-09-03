import click
import torch
import logging


def init_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def init_device():
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'
    return device
