"""Shared file recognition for viame.open and the inspect tool.

Recognition reads headers; integrity checks remain in inspect and the readers.
"""
import os

VIDEO_EXTS = (
    "3qp;3g2;amv;asf;avi;drc;gif;gifv;f4v;f4p;f4a;f4bflv;m4v;mkv;mp4;m4p;"
    "mpg;mpg2;mp2;mpeg;mpe;mpv;mng;mts;m2ts;mov;mxf;nsv;ogg;ogv;qt;roq;rm;"
    "rmvb;svi;webm;wmv;vob;yuv").split(';')
IMAGE_EXTS = (
    "webp;bmp;dds;gif;heic;jpg;jpeg;png;psd;psp;pspimage;tga;thm;tif;tiff;"
    "yuv").split(';')
MODEL_EXTS = ('pt', 'pth', 'ckpt', 'weights', 'onnx', 'zip')

# Magic numbers of the still image formats VIAME's readers accept
IMAGE_MAGIC = [
    (b'\xff\xd8\xff', 'JPEG', ('jpg', 'jpeg', 'thm')),
    (b'\x89PNG\r\n\x1a\n', 'PNG', ('png',)),
    (b'GIF87a', 'GIF', ('gif',)),
    (b'GIF89a', 'GIF', ('gif',)),
    (b'BM', 'BMP', ('bmp',)),
    (b'II*\x00', 'TIFF', ('tif', 'tiff')),
    (b'MM\x00*', 'TIFF', ('tif', 'tiff')),
    (b'DDS ', 'DDS', ('dds',)),
    (b'8BPS', 'Photoshop', ('psd',)),
]


def ext_of(path):
    return os.path.splitext(path)[1].lower().lstrip('.')


def read_head(path, n=16):
    try:
        with open(path, 'rb') as f:
            return f.read(n)
    except OSError:
        return b''


def sniff_image_magic(head):
    for magic, name, exts in IMAGE_MAGIC:
        if head.startswith(magic):
            return name, exts
    if head[:4] == b'RIFF' and head[8:12] == b'WEBP':
        return 'WebP', ('webp',)
    return None, ()


def sniff_video(head):
    if head[4:8] in (b'ftyp', b'moov', b'mdat'):
        return 'MP4/QuickTime'
    if head.startswith(b'\x1a\x45\xdf\xa3'):
        return 'Matroska/WebM'
    if head.startswith(b'RIFF') and head[8:12] == b'AVI ':
        return 'AVI'
    if head.startswith(b'\x30\x26\xb2\x75'):
        return 'ASF/WMV'
    if head.startswith(b'\x00\x00\x01\xba') or head.startswith(b'\x00\x00\x01\xb3'):
        return 'MPEG program stream'
    if head.startswith(b'FLV'):
        return 'FLV'
    if head.startswith(b'OggS'):
        return 'Ogg'
    return None



def file_kind(path):
    if os.path.isdir(path):
        return "directory"
    ext = ext_of(path)
    if ext in ("pipe", "conf", "csv", "json"):
        return ext
    if ext in MODEL_EXTS:
        return "model"
    head = read_head(path)
    if ext in IMAGE_EXTS or sniff_image_magic(head)[0]:
        return "image"
    if ext in VIDEO_EXTS or sniff_video(head):
        return "video"
    if ext == "txt":
        return "image_list"
    return "unknown"


def json_annotation_format(doc):
    if isinstance(doc, dict) and isinstance(doc.get("tracks"), dict):
        return "dive"
    if isinstance(doc, dict) and "images" in doc and "annotations" in doc:
        return "coco"
    return None


def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_viame_csv_row(columns):
    return len(columns) >= 9 and all(is_number(c) for c in columns[2:9])


def image_list_entries(path, lines):
    """Resolve paths relative to the list, with a working-directory fallback."""
    base = os.path.dirname(os.path.abspath(path))
    entries = []
    for line in lines:
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        relative = os.path.join(base, entry)
        entries.append(relative if os.path.isfile(relative) else os.path.abspath(entry))
    return entries
