from .device import (get_device_type,set_device_type,get_device_count,
                     set_device,to_device,stream,set_device_manual_seed,
                     data_to_device,current_stream,set_visible_devices,
                     init_stream, stream,get_device_info,
                     DEVICE_TYPE)

from .available import (is_cuda_available,is_npu_available,
                        is_mps_available,is_mlu_available)

from .memory import get_max_cuda_memory

__all__ = [
    'DEVICE_TYPE','init_stream', 'stream','get_device_info',
    'get_device_type', 'set_device_type',
    'get_device_count', 'set_device', 'to_device',
    'set_device_manual_seed', 'data_to_device',
    'current_stream','set_visible_devices',

    'is_cuda_available','is_npu_available','is_mps_available','is_mlu_available',

    'get_max_cuda_memory'
]

