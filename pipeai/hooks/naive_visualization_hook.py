from typing import Optional, Sequence

from pipeai.registry import HOOKS
from pipeai.hooks import Hook
from pipeai.hooks import DATA_BATCH


@HOOKS.register_module()
class NaiveVisualizationHook(Hook):
    """Show or Write the predicted results during the process of testing.

    Args:
        interval (int): Visualization interval. Defaults to 1.
        draw_gt (bool): Whether to draw the ground truth. Defaults to True.
        draw_pred (bool): Whether to draw the predicted result.
            Defaults to True.
    """
    priority = 'NORMAL'

    def __init__(self,
                 interval: int = 1,
                 draw_gt: bool = True,
                 draw_pred: bool = True):
        super().__init__()

        self._interval = interval
        self._draw_gt = draw_gt
        self._draw_pred = draw_pred

    def before_train(self, runner) -> None:
        """Call add_graph method of visualizer.

        Args:
            runner (Runner): The runner of the training process.
        """
        runner.visualizer.add_graph(runner.model, None)

    def after_test_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        outputs: Optional[Sequence] = None
    ) -> None:
        """Call visualizer to render task-specific samples.

        Args:
            runner (Runner): The training/testing runner.
            batch_idx (int): The index of the current batch.
            data_batch (dict | tuple | list, optional): Input data.
            outputs (Sequence, optional): Model predictions.
        """
        if self.every_n_inner_iters(batch_idx, self._interval):
            for data, output in zip(data_batch, outputs):  # type: ignore
                name = f"sample_{batch_idx}"
                runner.visualizer.add_datasample(
                    name,
                    data,       # raw input
                    data,       # ground truth (or extract from data['data_sample'])
                    output,     # prediction
                    self._draw_gt,
                    self._draw_pred,
                )
        # TODO: Hook 要做到 只负责调度，不做解析和绘制，任务相关的可视化逻辑必须交给 visualizer。
