import datetime

try:
    import torch.distributed as dist
except ModuleNotFoundError:
    dist = None

if dist is not None:
    _orig_init_pg = dist.init_process_group

    def _init_with_long_timeout(*args, **kwargs):
        kwargs.setdefault("timeout", datetime.timedelta(hours=10))
        return _orig_init_pg(*args, **kwargs)

    dist.init_process_group = _init_with_long_timeout
