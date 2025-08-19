from typing import Union, Sequence, Optional, Dict
from torch.utils.tensorboard import SummaryWriter

from .metric import AvgMeter


class Evaluator:
    """
    Simplified Evaluator for training/validation/testing stages.

    Manages scalar metrics (AvgMeter) with stage separation, logging, and TensorBoard plotting.
    """

    def __init__(self):
        from pipeai.logging import get_logger
        self.logger = get_logger(f"pipeai-{__name__}")

        self._dataset_meta: Optional[dict] = None
        self._pool: Dict[str, Dict] = {}

    def register(self, name: str, meter_type: str, fmt: str = '{:f}',
                 plt: bool = True, stage: str = 'train'):
        """
        Register a new metric.

        Args:
            name (str): Metric name (must be unique).
            meter_type (str): Metric type/category.
            fmt (str): Format string for printing.
            plt (bool): Whether to plot in TensorBoard.
            stage (str): Stage ('train', 'val', 'test').
        """
        if name in self._pool:
            raise ValueError(f"Metric {name} already exists.")

        self._pool[name] = {
            'meter': AvgMeter(),
            'index': len(self._pool),
            'format': fmt,
            'type': meter_type,
            'plt': plt,
            'stage': stage
        }

    def process(self, name: str, value: float, n: int = 1, stage: Optional[str] = None):
        """
        Update a metric.

        Args:
            name (str): Metric name.
            value (float): Value to update.
            n (int): Number of samples contributing.
            stage (Optional[str]): Stage to enforce (skip if mismatch).
        """
        if name not in self._pool:
            raise ValueError(f"Metric {name} not registered.")

        metric_stage = self._pool[name]['stage']
        if stage is not None and metric_stage != stage:
            self.logger.warning(
                f"Skipping update: metric '{name}' is registered for stage '{metric_stage}', not '{stage}'"
            )
            return

        self._pool[name]['meter'].update(value, n)

    def evaluate(self, stage: Optional[str] = None) -> Dict[str, float]:
        """
        Aggregate all metrics for the specified stage and return the final results as a dictionary.

        Args:
            stage (str, optional): Stage name (e.g., 'train', 'val', 'test').
                                   If None, metrics from all stages are returned.

        Returns:
            dict: {metric_name: metric_value}
        """
        results = {}
        for name, v in self._pool.items():
            if stage is None or v['stage'] == stage:
                results[name] = v['meter'].avg
        return results

    def get_avg(self, name: str, stage: Optional[str] = None) -> float:
        """
        Get average value of a metric.

        Args:
            name (str): Metric name.
            stage (Optional[str]): Stage check (optional).
        """
        if stage and self._pool[name]['stage'] != stage:
            self.logger.warning(
                f"Metric '{name}' stage mismatch: expected '{stage}', got '{self._pool[name]['stage']}'"
            )
        return self._pool[name]['meter'].avg

    def get_last(self, name: str, stage: Optional[str] = None) -> float:
        """
        Get last value of a metric.

        Args:
            name (str): Metric name.
            stage (Optional[str]): Stage check (optional).
        """
        if stage and self._pool[name]['stage'] != stage:
            self.logger.warning(
                f"Metric '{name}' stage mismatch: expected '{stage}', got '{self._pool[name]['stage']}'"
            )
        return self._pool[name]['meter'].last

    def reset(self, stage: Optional[str] = None):
        """
        Reset all metrics or metrics of a specific stage.

        Args:
            stage (Optional[str]): Stage to reset. Reset all if None.
        """
        for v in self._pool.values():
            if stage is None or v['stage'] == stage:
                v['meter'].reset()

    def print_metrics(self, stage: str):
        """
        Print metrics for a specific stage.

        Args:
            stage (str): Stage name.
        """
        out_list = [
            f"{name}: {v['format'].format(v['meter'].avg)}"
            for name, v in self._pool.items() if v['stage'] == stage
        ]
        if out_list:
            self.logger.info(f"[{stage}] Metrics: " + ", ".join(out_list))
        else:
            self.logger.info(f"[{stage}] No metrics to display.")

    def plot_metrics(self, stage: str, step: int, writer: SummaryWriter, value_type: str = 'avg'):
        """
        Plot metrics in TensorBoard for a specific stage.

        Args:
            stage (str): Stage name.
            step (int): Global step.
            writer (SummaryWriter): TensorBoard writer.
            value_type (str): 'avg' or 'last'.
        """
        assert value_type in ['avg', 'last'], "value_type must be 'avg' or 'last'"

        for name, v in self._pool.items():
            if v['stage'] != stage or not v['plt']:
                continue
            val = v['meter'].avg if value_type == 'avg' else v['meter'].last
            writer.add_scalar(name, val, global_step=step)

        # Flush once after all metrics
        writer.flush()
