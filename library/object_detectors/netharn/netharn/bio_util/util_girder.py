import ubelt as ub
from os.path import exists
from os.path import join


def grabdata_girder(api_url, file_id, fname=None, dpath=None, hash_prefix=None,
                    hasher='sha512', appname='bioharn', api_key=None,
                    verbose=1):
    """
    Downloads and caches a file from girder.

    Requirements:
        pip install girder-client

        import xdev
        globals().update(xdev.get_func_kwargs(grabdata_girder))

    Ignore:
        >>> api_url = 'https://data.kitware.com/api/v1'
        >>> file_id = '5dd3eb8eaf2e2eed3508d604'
        >>> fpath = grabdata_girder(api_url, file_id)
        >>> assert basename(fpath) == 'deploy_FCNN116_rlyehkac_094_DORPKG_py27Compat.zip'
    """
    import os

    # Use the CLI version to get a progress bar
    if verbose:
        from girder_client.cli import GirderCli
        client = GirderCli(username=None, password=None, apiUrl=api_url)
    else:
        import girder_client
        client = girder_client.GirderClient(apiUrl=api_url)

    auth_info = {'api_key': api_key}
    if auth_info.get('api_key', None) is None:
        auth_info['api_key'] = os.environ.get('GIRDER_API_KEY', None)
    if auth_info.get('api_key', None) is not None:
        client.authenticate(apiKey=auth_info['api_key'])

    if dpath is None:
        dpath = ub.ensure_app_cache_dir(appname)

    file_info = client.getFile(file_id)

    if fname is None:
        fname = file_info['name']

    print('file_info = {!r}'.format(file_info))
    hash_value = file_info.get('sha512', None)
    if hasher == 'sha512':
        if hash_prefix:
            if not hash_value.startswith(hash_prefix):
                raise ValueError('Incorrect got={}, want={}'.format(
                    hash_prefix, hash_value))
    else:
        raise KeyError('Unsupported hasher: {}'.format(hasher))

    fpath = join(dpath, fname)
    try:
        stamp = ub.CacheStamp(fname + '.hash', dpath=dpath, cfgstr=hash_value)
    except Exception:
        stamp = ub.CacheStamp(fname + '.hash', dpath=dpath, depends=hash_value)
    if stamp.expired() or not exists(fpath):
        client.downloadFile(file_id, fpath)
        stamp.renew()
    return fpath
