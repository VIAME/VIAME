#ckwg +4
# Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
# KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
# Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.

'''
Issues:
  - need types as defined by map-tk for input ports.

'''

from sprokit.pipeline import process
import os.path


class kw_archive_writer_process(process.PythonProcess):
    def __init__(self, conf ):
        process.PythonProcess.__init__(self, conf)

        # declare our configuration items
        self.declare_configuration_key(
            'output_directory',
            '.',
            'The path for output files.')

        self.declare_configuration_key(
            'base_filename',
            'kw_archive',
            'Base filename (no extension)')

        self.declare_configuration_key(
            'separate_meta',
            'true',
            'Whether to write separate .meta file')

        self.declare_configuration_key(
            'mission_id',
            '',
            'Mission id to store in archive')

        self.declare_configuration_key(
            'stream_id',
            '',
            'Stream id to store in archive')

        self.declare_configuration_key(
            'compress_image',
            'true',
            "Whether to compress image data stored in archive");

        # create port flags
        flags = process.PortFlags()
        flags.add(self.flag_required)

        # create input ports
        self.declare_input_port( # example
            'input',
            'integer',
            flags,
            'Where numbers are read from.')

        self.declare_input_port( 'timestamp', # name
                                 'timestamp', # type
                                 flags,
                                 'Timestamp for current inputs')

        self.declare_input_port( 'image',
                                 'vil_image_view_byte',
                                 flags,
                                 'Input image')

        self.declare_input_port( 'src_to_ref_homography',
                                 'image_to_image_homography',
                                 flags,
                                 'Src to ref homography')

        self.declare_input_port( 'corner_points',
                                 'video_metadata',
                                 process.PortFlags(), #optional
                                 'Video metadata defining corner points')

        self.declare_input_port( 'world_units_per_pixel',
                                 'double',
                                 flags,
                                 'Meters per pixel')


    # ----------------------------------------------------------------
    def _configure(self):
        # extract configuration values into local storage
        true_str = ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']

        self.C_output_directory = self.config_value('output_directory')
        self.C_base_filename = self.config_value('base_filename')
        self.C_separate_meta = self.config_value('separate_meta') in true_str
        self.C_mission_id = self.config_value('mission_id')
        self.C_stream_id = self.config_value('stream_id')
        self.C_compress_image = self.config_value('compress_image') in true_str

        self.open()

        self._base_configure()


    # ----------------------------------------------------------------
    def _reset(self):
        self.close()

        self._base_reset()


    # ----------------------------------------------------------------
    def _step(self):
        self.write_frame()

        self._base_step()

    # ----------------------------------------------------------------
    def open(self):
        base = os.path.join( C_output_directory, C_base_filename )
        self.fd_index = open( base + '.index', 'w')
        # write header to index stream
        self.fd_index.write( "4\n")
        self.fd_index.write( base + ".data\n")
        if C_separate_meta :
            self.fd_index.write( base + ".meta\n")
            self.fd_meta = open( base + '.meta', 'wb')
        else:
            self.fd_index.write( "\n")
            self.fd_meta = None

        self.fd_index.write( C_mission_id + "\n")
        self.fd_index.write( C_stream_id + "\n")

        self.fd_data = open( base + '.data', 'wb')

        if self.C_compress_image :
            self.fd_data.write( 3 ) # version 3 file = compressed
        else:
            self.fd_data.write( 2 ) # version 2 file = not compressed

        if self.fd_meta :
            self.fd_meta.write( 2 ) # version number

        # note that writes to fd_data and fd_meta are
        # vsl_b_write() format - TBD


    # ----------------------------------------------------------------
    def close(self):
        self.fd_index.close()
        self.fd_data.close()
        if seld.fd_meta:
            self.fd_meta.close()

        self.fd_index = None
        self.fd_data = None
        self.fd_meta = None

    # ----------------------------------------------------------------
    def write_frame(self):
        """
        This also is a vsl binary write
        """
        ts = self.grab_value_from_port('timesatmp')
        image = self.grab_value_from_port('image')
        s2r = self.grab_value_from_port('src_to_ref_homography')
        meta = self.grab_value_from_port('video_metadata')
        gsd = self.grab_value_from_port('world_units_per_pixel')


        self._base_step()
