from .device import (get_device_type,set_device_type,get_device_count,
                     set_device,to_device,stream,set_device_manual_seed,
                     data_to_device,current_stream,set_visible_devices)

__all__ = [
    'get_device_type', 'set_device_type',
    'get_device_count', 'set_device', 'to_device',
    'set_device_manual_seed', 'stream','data_to_device',
    'current_stream','set_visible_devices'
]

