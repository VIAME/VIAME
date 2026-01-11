"""
References:
    https://viame.readthedocs.io/en/latest/section_links/detection_file_conversions.html
"""
import ubelt as ub


class HabcamCSV(ub.NiceRepr):
    """
    TODO:
        See dvc-repos/viame_dvc/dev/convert_habcam_2014.py for other
        logic to convert to coco, move that here if this is needed more
        generally
    """
    def __init__(self, fpath):
        self.fpath = fpath

    def __nice__(self):
        return self.fpath

    def iter_raw_rows(self):
        with open(self.fpath, 'r') as file:
            line_iter1 = file.readlines()
            line_iter2 = (line for line in line_iter1)
            line_iter3 = (
                line for line in line_iter2
                if line and not line.startswith('#')
            )
            for idx, line in enumerate(line_iter3):
                # Skip header line
                if idx == 0 and line.startswith('"","X.2"'):
                    continue
                row = _parse_habcam_csv_line(line)
                yield row

    def iter_rows(self):
        for raw_row in self.iter_raw_rows():
            img = {
                'file_name': raw_row['imagename'] ,
                'raw_file_name': raw_row['Imagename'],
                'timestamp': raw_row['timestamp'],
            }
            cat = {
                'class_id': raw_row['class_id'],
                'category_id': raw_row['category_id'],
            }
            ann = {
                'category_id': raw_row['category_id'],
                'image_id': raw_row['image_id'],
                'annotation_id': raw_row['annotation_id'],

                'comment': raw_row['comment'],
                'source': raw_row['source'],
                'percent_cover': raw_row['percent_cover'],
                'geometry_id': raw_row['geometry_id'],
                'assignment_id': raw_row['assignment_id'],
                'annotator_id': raw_row['annotator_id'],
                'thegeom': raw_row['thegeom'],
                'geometry': raw_row['geometry'],
                'scope_id': raw_row['scope_id'],
                'X': raw_row['X'],
                'X.1': raw_row['X.1'],
                'X.2': raw_row['X.2'],
                'index': raw_row['index'],
            }
            row = {
                'cat': cat,
                'img': img,
                'ann': ann,
            }
            yield row


def _parse_habcam_csv_line(line):
    """
    Notes:
        Columns in Habcam CSV are:
            0: "",
            1: "X.2",
            2: "X.1",
            3: "X",
            4: "annotation_id",
            5: "image_id",
            6: "scope_id",
            7: "category_id",
            8: "geometry_text",
            9: "thegeom",
            10: "annotator_id",
            11: "assignment_id",
            12: "timestamp",
            13: "class_id",
            14: "deprecated",
            15: "geometry_id",
            16: "imagename",
            17: "assignment_num",
            18: "percent_cover",
            19: "comment",
            20: "source",
            21: "Imagename"
    """
    # For example:
    # Break the line down into pieces small pieces and then recombine pieces
    # where commas actually belong in the column.
    subparts = line.rstrip('\n').split(',')
    raw_columns = []

    braket_balance = 0
    next_subpart = []
    for subpart in subparts:
        braket_balance += subpart.count('[') - subpart.count(']')
        next_subpart.append(subpart)
        if braket_balance == 0:
            next_part = ','.join(next_subpart)
            raw_columns.append(next_part)
            next_subpart = []

    assert braket_balance == 0
    assert len(raw_columns) == 22

    def stripquote(x):
        if x.startswith('"') and x.endswith('"'):
            x = x[1:-1]
        return x

    def int_or_None(x):
        return None if x == 'NA' else int(x)

    columns = list(map(stripquote, raw_columns))
    raw_row = {
        'index': int(columns[0]),
        'X.2': float(columns[1]),
        'X.1': float(columns[2]),
        'X': float(columns[3]),
        'annotation_id': columns[4],
        'image_id': columns[5],
        'scope_id': int_or_None(columns[6]),
        'category_id': int(columns[7]),
        'geometry': parse_geometry_text(columns[8]),
        'thegeom': columns[9],
        'annotator_id': columns[10],
        'assignment_id': int(columns[11]),
        'timestamp': columns[12],
        'class_id': int(columns[13]),
        'deprecated': columns[14],
        'geometry_id': columns[15],
        'imagename': columns[16],
        'assignment_num': int(columns[17]),
        'percent_cover': columns[18],
        'comment': columns[19],
        'source': columns[20],
        'Imagename': columns[21],
    }
    return raw_row


def parse_geometry_text(text):
    """
    Example:
        >>> text = '""boundingBox"": [[886.6666666666666, 578.6666666666666], [1028, 734.6666666666666]]'
        >>> print(parse_geometry_text(text))
        >>> text = 'whole image'
        >>> print(parse_geometry_text(text))
        >>> text = '""line"": [[1344.0185165405273, 866.240732828776], [1074.685183207194, 564.9073994954427]]'
        >>> print(parse_geometry_text(text))
    """
    import json
    geom_type, *geom_data = text.split(':')
    geom_data = ':'.join(geom_data).strip()
    geom = {
        'type': geom_type.replace('"', '').strip('')
    }
    if geom_data:
        geom['data'] = json.loads(geom_data)
    return geom


def _demo_habcam_csv_text():
    text = ub.codeblock('''
    "1",1,14,6000,"ann_9bfa19cf03697345d8294b3321266169122feda1","201403.20140704.055257707.310400.jpg",1,1003,"""boundingBox"": [[420, 201.33333333333334], [600, 354.6666666666667]]","010300000001000000050000000000000000407A40ABAAAAAAAA2A69400000000000407A40ABAAAAAAAA2A76400000000000C08240ABAAAAAAAA2A76400000000000C08240ABAAAAAAAA2A69400000000000407A40ABAAAAAAAA2A6940","bshank",201499,"2014-07-21 15:13:04-04",1003,"f",NA,"201403.20140704.055257707.310400.png",201499,NA,NA,"","201403.20140704.055257707.310400.tif"
    "2",2,15,6235,"ann_6e262901fa318c9487af27c802b2c9edf08a08b4","201403.20140703.234215605.155300.jpg",1,1003,"""boundingBox"": [[886.6666666666666, 578.6666666666666], [1028, 734.6666666666666]]","010300000001000000050000005555555555B58B4055555555551582405555555555B58B405555555555F5864000000000001090405555555555F58640000000000010904055555555551582405555555555B58B405555555555158240","achute",201499,"2014-07-21 12:17:21-04",1003,"f",NA,"201403.20140703.234215605.155300.png",201499,NA,NA,"","201403.20140703.234215605.155300.tif"
    ''')
    line = text.split('\n')[0]
