
if __name__ == '__main__':
    r"""
    CommandLine:

        workon_py2
        source ~/code/VIAME/build-py2.7/install/setup_viame.sh

        python ~/code/VIAME/plugins/camtrawl/python/camtrawl/demo.py

        ffmpeg -y -f image2 -i out_haul83/%*.png -vcodec mpeg4 -vf "setpts=10*PTS" haul83-results.avi
    """
    try:
        from camtrawl import demo
    except ImportError:
        import warnings
        warnings.warn('Camtrawl main not in PYTHONPATH. Fixing')
        from os.path import dirname
        import sys
        sys.path.append(dirname(dirname(__file__)))
        from camtrawl import demo

    demo.setup_demo_logger()
    measurements = demo.demo()
