from .registry import Registry


# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry('loop')
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry('hook')

# manage logprocessor
LOG_PROCESSORS = Registry('log_processor')

# manage all kinds of metrics
METRICS = Registry('metric')
