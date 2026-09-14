"""
DEPRECATED. Moved to kwcoco.data
"""


def main():
    """
    Dump the paths to the coco file to stdout

    By default these will go to in the path:
        ~/.cache/netharn/camvid/camvid-master

    The four files will be:
        ~/.cache/kwcoco/camvid/camvid-master/camvid-full.mscoco.json
        ~/.cache/kwcoco/camvid/camvid-master/camvid-train.mscoco.json
        ~/.cache/kwcoco/camvid/camvid-master/camvid-vali.mscoco.json
        ~/.cache/kwcoco/camvid/camvid-master/camvid-test.mscoco.json
    """
    from kwcoco.data import grab_camvid
    return grab_camvid.main()

if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.data.grab_camvid
    """
    main()
