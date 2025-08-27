import warnings
from typing import Optional, Union, Dict, Tuple

from numpy import inf, isfinite

from .hook import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]


class EarlyStoppingHook(Hook):
    """Early stops training when a monitored metric has stopped improving.

    This hook checks a specified metric at the end of each validation epoch.
    If the metric does not improve by at least `min_delta` for a given
    number of epochs (`patience`), the training process is stopped.

    Args:
        monitor (str): The name of the metric to monitor (e.g., 'val/loss').
        rule (Optional[str]): The comparison rule. Can be 'greater' or 'less'.
            If None, the rule is inferred from the metric name. Defaults to None.
        min_delta (float): The minimum change in the monitored metric to be
            considered an improvement. Defaults to 0.1.
        strict (bool): Whether to raise an error if the monitored metric is
            not found in the validation results. Defaults to False.
        check_finite (bool): Whether to stop training if the monitored metric
            becomes NaN or infinity. Defaults to True.
        patience (int): The number of epochs with no improvement after which
            training will be stopped. Defaults to 5.
        stopping_threshold (Optional[float]): An optional value. If the metric
            reaches this threshold, training will stop immediately.
            Defaults to None.
    """

    priority = 'LOWEST'

    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    _default_greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc'
    ]
    _default_less_keys = ['loss']

    def __init__(self,
                 monitor: str,
                 rule: Optional[str] = None,
                 min_delta: float = 0.1,
                 strict: bool = False,
                 check_finite: bool = True,
                 patience: int = 5,
                 stopping_threshold: Optional[float] = None, ):
        super().__init__()

        self.monitor = monitor
        if rule is not None:
            if rule not in ['greater', 'less']:
                raise ValueError(
                    '`rule` should be either "greater" or "less", '
                    f'but got {rule}')
        else:
            rule = self._init_rule(monitor)
        self.rule = rule  # type: ignore

        self.min_delta = min_delta if rule == 'greater' else -1 * min_delta
        self.strict = strict
        self.check_finite = check_finite
        self.patience = patience
        self.stopping_threshold = stopping_threshold  # type: ignore
        self.wait_count = 0
        self.best_score = -inf if rule == 'greater' else inf

    def before_run(self, runner) -> None:
        """Verifies the runner's training loop has the `stop_training` attribute.

        Args:
            runner (Runner): The runner of the training process. It is expected
                to have a `train_loop` attribute with a `stop_training` flag.
        """
        assert hasattr(runner.train_loop, 'stop_training'), \
            '`train_loop` should contain `stop_training` variable.'

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """Checks the monitored metric after a validation epoch to decide on stopping.

        This method retrieves the monitored metric, evaluates the stopping
        conditions, and sets the runner's `stop_training` flag if necessary.

        Args:
            runner (Runner): The runner of the training process.
            metrics (Optional[Dict[str, float]]): A dictionary of evaluation
                results from the validation epoch.
        """
        if self.monitor not in metrics:
            if self.strict:
                raise RuntimeError(
                    'Early stopping conditioned on metric '
                    f'`{self.monitor}` is not available. Please check available'
                    f' metrics {metrics.keys()}, or set `strict=False` in '
                    '`EarlyStoppingHook`.')
            warnings.warn(
                'Skip early stopping process since the evaluation '
                f'results ({metrics.keys()}) do not include `monitor` '
                f'({self.monitor}).')
            return

        current_score = metrics[self.monitor]
        stop_training, message = self._check_stop_condition(current_score)

        if stop_training:
            runner.train_loop.stop_training = True
            runner.logger.info(message)

    def _init_rule(self, monitor: str) -> str:
        """Infers the comparison rule ('greater' or 'less') from the monitor's name.

        Args:
            monitor (str): The name of the metric.

        Returns:
            str: The inferred rule, either 'greater' or 'less'.
        """
        greater_keys = {key.lower() for key in self._default_greater_keys}
        less_keys = {key.lower() for key in self._default_less_keys}
        monitor_lc = monitor.lower()

        if monitor_lc in greater_keys:
            rule = 'greater'
        elif monitor_lc in less_keys:
            rule = 'less'
        elif any(key in monitor_lc for key in greater_keys):
            rule = 'greater'
        elif any(key in monitor_lc for key in less_keys):
            rule = 'less'
        else:
            raise ValueError(f'Cannot infer the rule for {monitor}, thus rule '
                             'must be specified.')
        return rule

    def _check_stop_condition(self, current_score: float) -> Tuple[bool, str]:
        """Evaluates the current score against all stopping criteria.

        Checks for three conditions in order:
        1. The score is not finite (NaN or infinity).
        2. The score has reached the `stopping_threshold`.
        3. The score has not improved for `patience` epochs.

        Args:
            current_score (float): The latest value of the monitored metric.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether
            to stop training and a message explaining the reason.
        """
        compare = self.rule_map[self.rule]
        stop_training = False
        reason_message = ''

        if self.check_finite and not isfinite(current_score):
            stop_training = True
            reason_message = (f'Monitored metric {self.monitor} = '
                              f'{current_score} is infinite. '
                              f'Previous best value was '
                              f'{self.best_score:.3f}.')

        if self.stopping_threshold is not None and compare(
                current_score, self.stopping_threshold):
            stop_training = True
            self.best_score = current_score
            reason_message = (f'Stopping threshold reached: '
                              f'`{self.monitor}` = {current_score} is '
                              f'{self.rule} than {self.stopping_threshold}.')

        if compare(self.best_score + self.min_delta, current_score):
            self.wait_count += 1
            if self.wait_count >= self.patience:
                reason_message = (f'the monitored metric did not improve '
                                  f'in the last {self.wait_count} records. '
                                  f'best score: {self.best_score:.3f}. ')
                stop_training = True
        else:
            self.best_score = current_score
            self.wait_count = 0

        return stop_training, reason_message
