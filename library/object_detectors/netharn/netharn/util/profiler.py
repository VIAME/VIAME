try:
    # Netharn used to have a profiler. Now its moved to xdev
    from xdev import profiler
    profile = profiler.profile
    profile_now = profiler.profile_now
    IS_PROFILING = profiler.IS_PROFILING
except ImportError:
    def __dummy_profile__(func):
        """ dummy profiling func. does nothing """
        return func
    profile = __dummy_profile__
    profile_now = __dummy_profile__
    IS_PROFILING = False
