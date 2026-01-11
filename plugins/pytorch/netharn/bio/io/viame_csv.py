from os.path import exists
from os.path import join
import ubelt as ub


class ViameCSV(ub.NiceRepr):
    """
    Basic script to convert VIAME-CSV to kwcoco

    References:
        https://viame.readthedocs.io/en/latest/section_links/detection_file_conversions.html

    TODO:
        implement load
        ~/code/bioharn/dev/data_tools/kwcoco_to_viame_csv.py

    Example:
        >>> from .io.viame_csv import *
        >>> fpath = demodata_viame_csv_fpath()
        >>> self = ViameCSV(fpath)
        >>> list(self.iter_rows())
        >>> dset = self.to_coco()
        >>> print('dset = {!r}'.format(dset))
    """
    def __init__(self, fpath):
        self.fpath = fpath
        self.rows = None

    def __nice__(self):
        return self.fpath

    def iter_rows(self):
        with open(self.fpath, 'r') as file:
            line_iter1 = file.readlines()
            line_iter2 = (line for line in line_iter1)
            line_iter3 = (
                line for line in line_iter2
                if line and not line.startswith('#')
            )
            for line in line_iter3:
                row = _parse_viame_csv_line(line)
                yield row

    def to_coco(self):
        """
        Converts to a coco dataset
        """
        import kwcoco
        dset = kwcoco.CocoDataset()
        dset = self.extend_coco(dset)
        return dset

    def extend_coco(self, dset):
        """
        Extends an existing coco dataset with information from this CSV
        """
        for row in self.iter_rows():
            cat = row['cat']
            img = row['img']
            ann = row['ann']
            cid = dset.ensure_category(**cat)
            gid = dset.ensure_image(**img)
            ann.update({
                'image_id': gid,
                'category_id': cid,
            })
            dset.add_annotation(**ann)
        return dset


def _parse_viame_csv_line(line):
    """
    Notes:
        Columns in a VIAME CSV are:

        - 1: Detection or Track Unique ID
        - 2: Video or Image String Identifier
        - 3: Unique Frame Integer Identifier
        - 4: TL-x (top left of the image is the origin: 0,0)
        - 5: TL-y
        - 6: BR-x
        - 7: BR-y
        - 8: Auxiliary Confidence (how likely is this actually an object)
        - 9: Target Length

        Optional:

        - 10,11+ : class-name, score (this pair may be omitted or repeated)

    """
    parts = line.split(',')
    tid = parts[0]
    gname = parts[1]

    frame_index = parts[2]
    tl_x, tl_y, br_x, br_y = map(float, parts[3:7])
    w = br_x - tl_x
    h = br_y - tl_y
    bbox = [tl_x, tl_y, w, h]
    score = float(parts[7])
    target_len = float(parts[8])

    rest = parts[9:]
    catparts = []
    rest_iter = iter(rest)
    for p in rest_iter:
        if p.startswith('('):
            catparts.append(p)

    final_parts = list(rest_iter)
    if final_parts:
        raise NotImplementedError

    catnames = rest[0::2]
    catscores = list(map(float, rest[1::2]))

    cat_to_score = ub.dzip(catnames, catscores)
    catname = ub.argmax(cat_to_score)

    try:
        frame_index = int(frame_index)
    except Exception:
        pass

    cat = {
        'name': catname,
    }
    img = {
        'file_name': gname,
        'frame_index': frame_index,
    }
    ann = {
        'bbox': bbox,
        'score': score,
        'target_len': target_len,
        'track_id': int(tid),
    }
    row = {
        'cat': cat,
        'img': img,
        'ann': ann,
    }
    return row


def demodata_viame_csv_fpath():
    test_dpath = ub.ensure_app_cache_dir('viame/tests/')
    csv_fpath = join(test_dpath, 'demo_viame_csv.csv')
    if not exists(csv_fpath):
        text = ub.codeblock(
            '''
            1,,0,571,266,750,337,1,0,seriola,1
            1,,1,567,262,723,335,1,0,seriola,1
            1,,2,564,258,697,333,1,0,seriola,1
            1,,3,561,254,670,331,1,0,seriola,1
            1,,4,558,251,644,330,1,0,seriola,1
            1,,5,555,247,626,328,1,0,seriola,1
            1,,6,552,244,609,326,1,0,seriola,1
            1,,7,548,240,613,324,1,0,seriola,1
            1,,8,545,236,617,322,1,0,seriola,1
            1,,9,542,233,621,320,1,0,seriola,1
            1,,10,539,229,625,318,1,0,seriola,1
            1,,11,536,226,629,317,1,0,seriola,1
            1,,12,543,220,633,307,1,0,seriola,1
            1,,13,550,214,637,297,1,0,seriola,1
            1,,14,558,208,642,287,1,0,seriola,1
            1,,15,565,201,649,276,1,0,seriola,1
            1,,16,572,195,656,266,1,0,seriola,1
            1,,17,580,189,664,256,1,0,seriola,1
            1,,18,587,183,671,246,1,0,seriola,1
            1,,19,595,177,679,236,1,0,seriola,1
            1,,20,602,170,684,228,1,0,seriola,1
            1,,21,610,163,689,220,1,0,seriola,1
            1,,22,617,156,695,212,1,0,seriola,1
            1,,23,625,149,700,204,1,0,seriola,1
            1,,24,632,142,705,196,1,0,seriola,1
            1,,25,640,135,711,189,1,0,seriola,1
            1,,26,647,132,711,184,1,0,seriola,1
            1,,27,655,130,712,180,1,0,seriola,1
            1,,66,750,110,784,140,1,0,seriola,1
            1,,67,755,109,792,140,1,0,seriola,1
            1,,68,761,109,801,140,1,0,seriola,1
            1,,69,767,108,810,140,1,0,seriola,1
            1,,70,773,108,819,140,1,0,seriola,1
            1,,71,779,108,828,140,1,0,seriola,1
            1,,72,785,107,836,140,1,0,seriola,1
            1,,73,791,107,845,140,1,0,seriola,1
            1,,74,797,106,854,140,1,0,seriola,1
            1,,75,803,106,863,140,1,0,seriola,1
            1,,76,809,106,872,141,1,0,seriola,1
            1,,77,817,105,879,140,1,0,seriola,1
            1,,78,825,104,886,139,1,0,seriola,1
            1,,79,833,103,893,138,1,0,seriola,1
            1,,80,842,103,901,138,1,0,seriola,1
            1,,81,850,102,908,137,1,0,seriola,1
            1,,82,858,101,915,136,1,0,seriola,1
            1,,83,867,101,923,136,1,0,seriola,1
            2,,0,551,158,706,270,1,0,seriola,1
            2,,1,555,156,713,266,1,0,seriola,1
            2,,2,560,153,720,263,1,0,seriola,1
            2,,3,564,151,728,259,1,0,seriola,1
            2,,4,569,149,735,255,1,0,seriola,1
            2,,5,574,146,742,252,1,0,seriola,1
            2,,6,578,144,750,248,1,0,seriola,1
            2,,7,583,142,757,244,1,0,seriola,1
            2,,8,587,139,764,241,1,0,seriola,1
            2,,9,592,137,772,237,1,0,seriola,1
            ''')
        with open(csv_fpath, 'w') as file:
            file.write(text)
    return csv_fpath
