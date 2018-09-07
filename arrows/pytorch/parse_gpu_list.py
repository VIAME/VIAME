"""A small module to provide a consistent method for turning GPU list
argument strings to processes into a list of GPU indices and for
turning that into a Pytorch device.

"""

import torch

def parse_gpu_list(gpu_list_str):
    """Parse a string representing a list of GPU indices to a list of
    numeric GPU indices.  The indices should be separated by commas.
    Two special values are understood: the string "None" will produce
    an empty list, and the string "all" will produce the value None
    (which has a special meaning when picking a device).  Note that
    "None" is the only way to produce an empty list; an empty string
    won't work.

    """
    return ([] if gpu_list_str == 'None' else
            None if gpu_list_str == 'all' else
            list(map(int, gpu_list_str.split(','))))

def get_device(gpu_list=None):
    """Get a Pytorch device corresponding to one of the GPU indices listed
    in gpu_list.  If gpu_list is empty, get the device corresponding
    to the CPU instead.  If gpu_list is None (the default), enumerate
    the available GPU indices and pick one as though the list had been
    passed directly, except that in the case of there being no GPUs,
    an IndexError will be thrown.

    The return value is a pair of the device and a boolean that is
    true if the returned device is a GPU device.

    Note that we currently return the first listed device.

    """
    if gpu_list is None:
        gpu_list = list(range(torch.cuda.device_count()))
    elif not gpu_list:
        return torch.device('cpu'), False
    return torch.device('cuda:{}'.format(gpu_list[0])), True
