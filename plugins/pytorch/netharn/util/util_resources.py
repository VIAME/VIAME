def ensure_ulimit():
    """
    srun -c 4 -p priority python -c "import resource; print(resource.getrlimit(resource.RLIMIT_NOFILE))"
    """
    # NOTE: It is important to have a high enought ulimit for DataParallel
    try:
        import resource
        # RLIMIT_NOFILE specifies a value one greater than the maximum file
        # descriptor number that can be opened by this process
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        if rlimit[0] <= 8192:
            resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
    except Exception:
        print('Unable to fix ulimit. Ensure manually')
        raise


def resource_usage(workdir='.', disk=True, ram=True, rlimit=True, rusage=False):
    """
    Get information about resource usage

    Args:
        workdir (str): path to directory to check disk space of.

    Returns:
        Dict: dictionary of resource info

    References:
        https://stackoverflow.com/questions/276052/how-to-get-current-cpu-and-ram-usage-in-python
        https://man7.org/linux/man-pages/man2/getrlimit.2.html

    Example:
        >>> from .util.util_resources import *  # NOQA
        >>> info = resource_usage()
        >>> import ubelt as ub
        >>> print('info = {}'.format(ub.repr2(info, nl=2, precision=2)))

    Profiling:
        import timerit
        ti = timerit.Timerit(100, bestof=10, verbose=2)
        for timer in ti.reset('resource_usage'):
            with timer:
                resource_usage()

        import xdev
        _  = xdev.profile_now(resource_usage)()
    """

    def struct_to_dict(struct):
        attrs = {k for k in dir(struct) if not k.startswith('_')}
        dict_ =  {k: getattr(struct, k) for k in attrs}
        dict_ = {k: v for k, v in dict_.items() if isinstance(v, (int, float))}
        return dict_

    errors = []
    info = {}
    try:
        import psutil
    except Exception as ex:
        errors.append(repr(ex))
    else:
        if ram:
            vmem = psutil.virtual_memory()
            # info['cpu_percent'] = psutil.cpu_percent()
            info['mem_details'] = dict(vmem._asdict())
            info['ram_percent'] = vmem.percent
            info['mem_percent'] = round(vmem.used * 100 / vmem.total, 2)

    if disk:
        import shutil
        disk_usage = shutil.disk_usage(workdir)
        info['disk_usage'] = struct_to_dict(disk_usage)

    try:
        import resource
    except Exception as ex:
        errors.append(repr(ex))
    else:
        if rlimit:
            rlimit_keys = [
                # 'RLIMIT_AS',
                # 'RLIMIT_CORE',
                # 'RLIMIT_CPU',
                # 'RLIMIT_DATA',
                # 'RLIMIT_FSIZE',

                # 'RLIMIT_MEMLOCK',
                # 'RLIMIT_MSGQUEUE',

                # 'RLIMIT_NICE',
                # RLIMIT_NOFILE - a value one greater than the maximum file
                # descriptor number that can be opened by this process
                'RLIMIT_NOFILE',

                # 'RLIMIT_NPROC',

                # 'RLIMIT_OFILE',

                # 'RLIMIT_RSS',

                # 'RLIMIT_RTPRIO',
                # 'RLIMIT_SIGPENDING',
                # 'RLIMIT_STACK',
            ]
            rlimits = {}
            for key in rlimit_keys:
                # each rlimit return a (soft, hard) tuple with soft and hard limits
                rlimits[key] = resource.getrlimit(getattr(resource, key))
            info['rlimits'] = rlimits

        if rusage:
            rusage_keys = [
                # 'RUSAGE_CHILDREN',
                'RUSAGE_SELF',
                # 'RUSAGE_THREAD',
            ]
            rusages = {}
            for key in rusage_keys:
                # Returns a structure that we will convert to a dict
                val = resource.getrusage(getattr(resource, key))
                attrs = (
                    {n for n in dir(val) if not n.startswith('_')} -
                    {'count', 'index'}
                )
                val = {n: getattr(val, n) for n in attrs}
                rusages[key] = val
            info['rusages'] = rusages

    if errors:
        info['errors'] = errors

    return info
