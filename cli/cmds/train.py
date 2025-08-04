import os
import sys

import typer

sys.path.insert(0, os.getcwd())


def run(
        cfg: str = typer.Option(..., "--cfg", "-c", help="Path to config file"),
        devices: str = typer.Option(None, help="CUDA_VISIBLE_DEVICES"),
        node_rank: int = typer.Option(0, help="Distributed node rank")
):
    from moirai import launch_training
    launch_training(cfg, devices, node_rank)
