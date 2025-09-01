import torch
from torch.profiler import profile, record_function, ProfilerActivity

from .base import Hook


class ProfilerHook(Hook):
    """A training hook that integrates torch.profiler for performance profiling.

    Args:
        activities (list[ProfilerActivity], optional): The list of activities to profile.
            Defaults to CPU and CUDA if available.
        schedule (Callable, optional): The schedule function that controls
            when to start/stop profiling. Defaults to None.
        on_trace_ready (Callable, optional): A function called at the end of each profiling step.
            Defaults to None.
        record_shapes (bool): Whether to record operator input shapes. Defaults to True.
        profile_memory (bool): Whether to track memory usage. Defaults to True.
        with_stack (bool): Whether to record the source code stacks. Defaults to False.
        with_flops (bool): Whether to estimate FLOPs of operators. Defaults to False.
    """

    def __init__(self,
                 activities=None,
                 schedule=None,
                 on_trace_ready=None,
                 record_shapes=True,
                 profile_memory=True,
                 with_stack=False,
                 with_flops=False):
        super().__init__()

        self.activities = activities or [
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ]
        self.schedule = schedule
        self.on_trace_ready = on_trace_ready
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops

        self.profiler = None

    def before_train(self, runner):
        """Initialize the profiler before training begins.

        Args:
            runner (Runner): The training runner instance.
        """
        self.profiler = profile(
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=self.on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops
        )
        self.profiler.__enter__()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Step the profiler after each training iteration.

        Args:
            runner (Runner): The training runner instance.
            batch_idx (int): The current batch index.
            data_batch (Any, optional): The input batch. Defaults to None.
            outputs (Any, optional): The model outputs. Defaults to None.
        """
        if self.profiler is not None:
            self.profiler.step()

    def after_train(self, runner):
        """Close the profiler after training ends.

        Args:
            runner (Runner): The training runner instance.
        """
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
