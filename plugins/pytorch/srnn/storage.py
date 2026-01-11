# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Abstract access to storage for training data"""
import numpy
from pathlib import Path as _Path
import sqlite3


class VideoId:
    """Collection of information sufficing to identify a video in a
    DataStorage

    """
    __slots__ = '_id',

    def __init__(self, id):
        """Private.  Use DataStorage.video_id to create instances"""
        self._id = id

    def __eq__(self, other):
        if not isinstance(other, VideoId):
            return NotImplemented
        return self._id == other._id

    def __hash__(self):
        return hash((VideoId, self._id))


class DetectionId:
    """Collection of information sufficing to identify a detection in a
    DataStorage

    """
    __slots__ = '_id',

    def __init__(self, id):
        """Private.  Use DataStorage.detection_id to create instances"""
        self._id = id


class SequenceList:
    """Random-access sequence of "sequences" -- lists of DetectionIds

    """
    __slots__ = '_starts', '_det_ids'

    def __init__(self, starts, det_ids):
        """Private.  Use .from_iterable to create instances"""
        self._starts = starts
        self._det_ids = det_ids

    @classmethod
    def from_iterable(cls, sequences):
        """Create a SequenceList from an iterable of (variable-length)
        iterables of DetectionIds

        """
        det_ids = []
        starts = [0]
        for dets in sequences:
            det_ids.extend(det._id for det in dets)
            starts.append(len(det_ids))
        return cls(numpy.array(starts), numpy.array(det_ids))

    def __len__(self):
        return len(self._starts) - 1

    def __getitem__(self, x):
        """Get the sequence at the given index.  Slices are not supported"""
        if x < 0:
            x += len(self)
            if x < 0:
                raise IndexError
        det_ids = self._det_ids[self._starts[x]:self._starts[x + 1]]
        # _id has a numpy integer type, so we convert it to the normal
        # int type for sqlite3
        return [DetectionId(int(_id)) for _id in det_ids]


class DataStorage:
    __slots__ = '_con_', '_root', '_auto_flush_counter'
    _AUTO_FLUSH_THRESHOLD = 10_000

    def __init__(self, root):
        """Create a DataStorage backed by a directory tree rooted at root"""
        self._root = root
        self._con_ = None
        self._auto_flush_counter = 0

    def video_id(self, name, phase):
        """Create a VideoId with the following components:
        - name: string identifier
        - phase: "train" or "test"

        Note: Two calls to this function with the same parameters will
        return unequal results.

        """
        row = name, phase
        _id = self._con.execute(_CREATE_VIDEO_ID, row).lastrowid
        return VideoId(_id)

    def detection_id(
            self, video_id, track_id, frame_id,
    ):
        """Create a DetectionId with the following components:
        - video_id: VideoId
        - track_id: integer track ID, unique per video
        - frame_id: integer frame ID, unique per video

        Note: Two calls to this function with the same parameters will
        return unequal results.

        """
        row = video_id._id, track_id, frame_id
        _id = self._con.execute(_CREATE_DETECTION_ID, row).lastrowid
        return DetectionId(_id)

    def blob(self, detection_id, feature):
        """Return a DataStorage.Blob object that can be used to access the
        desired data

        Arguments:
        - detection_id: a DetectionId
        - feature: a string identifying what type of data (e.g. a
          particular feature vector) the blob contains
        """
        return self.Blob._from_parts(self, detection_id, feature)

    class Blob:
        """Handle for reading and writing arbitrary binary data in a
        DataStorage

        """
        __slots__ = '_ds', '_det_id', '_feature'

        def __init__(self, data_storage, detection_id, feature):
            """Private.  Use DataStorage.blob to create instances"""
            self._ds = data_storage
            self._det_id = detection_id
            self._feature = feature

        @classmethod
        def _from_parts(cls, data_storage, detection_id, feature):
            return cls(data_storage, detection_id, feature)

        def read(self):
            """Return a bytes of the data associated with this Blob if it exists,
            else raise

            """
            args = self._det_id._id, self._feature
            [[data]] = self._ds._con.execute(_READ_BLOB, args)
            return data

        def write(self, data):
            """Set the data of this Blob to the given bytes

            Note: A Blob's data may only be set once.

            """
            args = self._det_id._id, self._feature, data
            self._ds._con.execute(_CREATE_BLOB, args)
            self._ds._written()

    def _written(self):
        """Increment the write counter, automatically flushing if necessary"""
        afc = self._auto_flush_counter + 1
        if afc >= self._AUTO_FLUSH_THRESHOLD:
            self.flush()
            afc = 0
        self._auto_flush_counter = afc

    @property
    def _con(self):
        """Get the database connection underlying this DataStorage

        This is a property in order to allow auto-initialization of
        the connection, which is necessary for multi-process use with
        PyTorch's DataLoader because:
        - sqlite3 connections don't work across fork()
          (https://sqlite.org/faq.html#q6)
        - There is currently (as of PyTorch 1.4.0) no way to have a
          function called on the Dataset after forking
          (cf. https://github.com/pytorch/pytorch/issues/13023)

        """
        if self._con_ is None:
            self._con_ = sqlite3.connect(_Path(self._root, 'db.sqlite'))
        return self._con_

    def create(self):
        """Initialize underlying storage.  This can safely be called multiple
        times.

        """
        self._con.executescript(_CREATE_TABLES)

    def flush(self):
        """Flush any pendings writes to this DataStorage"""
        con = self._con_
        if con is not None:
            con.commit()

    def close(self):
        """Close any open resources for this DataStorage.  They will be
        reopened as needed.

        This methods should be called before forking if the underlying
        storage has been accessed.

        """
        self.flush()
        con = self._con_
        if con is not None:
            con.close()
            self._con_ = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        """Make a last-ditch effort to properly clean up"""
        self.close()


_CREATE_TABLES = """
pragma journal_mode=wal;

create table if not exists "videos" (
    "id" integer primary key,
    "name" text,
    "phase" text
);

create table if not exists "detections" (
    "id" integer primary key,
    "video_id" integer,
    "track_id" integer,
    "frame_id" integer,
    foreign key ("video_id") references "videos" ("id")
);

create table if not exists "blobs" (
    "feature" text,
    "det_id" integer,
    "data" blob,
    primary key ("feature", "det_id"),
    foreign key ("det_id") references "detections" ("id")
);
"""

_CREATE_VIDEO_ID = 'insert into "videos" ("name", "phase") values (?, ?)'

_CREATE_DETECTION_ID = """
insert into "detections"
("video_id", "track_id", "frame_id")
values (?, ?, ?)
"""

_READ_BLOB = 'select "data" from "blobs" where "det_id" = ? and "feature" = ?'
_CREATE_BLOB = """
insert into "blobs"
("det_id", "feature", "data")
values (?, ?, ?)
"""
