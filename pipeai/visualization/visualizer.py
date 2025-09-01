from typing import Optional, Union, List, Dict

import numpy as np

from pipeai.registry import VISUALIZERS

VisBackendsType = Union[List[Union[List]], dict, None]

@VISUALIZERS.register_module()
class Visualizer():
    """PIPEAI provides a Visualizer class that uses the ``Matplotlib``
    library as the backend.

    """

    def __init__(self,
                 name='visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: VisBackendsType = None,
                 save_dir: Optional[str] = None,
                 fig_save_cfg=dict(frameon=False),
                 fig_show_cfg=dict(frameon=False)) -> None:
        super().__init__()

        self._dataset_meta: Optional[dict] = None
        self._vis_backends: Dict[str, BaseVisBackend] = {}
        pass
