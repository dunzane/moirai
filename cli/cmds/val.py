from typing import Optional

import typer

from moirai import Config, Runner, launch_runner

val_app = typer.Typer()


@val_app.command(name="val")
def run(
    cfg: str = typer.Option(..., "--cfg", "-c", help="Path to config file"),
    ckpt: Optional[str] = typer.Option(None, "--ckpt", help="Checkpoint path"),
    device_type: str = typer.Option("gpu", "--device-type", help="Device type"),
    devices: Optional[str] = typer.Option(None, "--devices", help="CUDA_VISIBLE_DEVICES"),
):
    def main(cfg_dict, runner: Runner, ckpt_path: Optional[str] = None):
        runner.build_logger(logger_name='moirai-inference', log_file_name='validate_result')
        runner.load_model(ckpt_path=ckpt_path)
        runner.validate(cfg_dict)

    launch_runner(cfg, main, args=(ckpt,), device_type=device_type, devices=devices)
