"""Adapt normal pipeline graphs to the native in-memory adapters."""
import math
from pathlib import Path
import tempfile
import threading
import time

from ._io import Pipeline, _algorithms, _pipeline_root

def _prepare(path, inputs, outputs):
    from viame.core.embedded_pipeline import prepare_embedded_pipeline
    return prepare_embedded_pipeline(
        str(path), search_paths=[str(_pipeline_root())],
        inputs=[inputs] if isinstance(inputs, str) else inputs,
        outputs=[outputs] if isinstance(outputs, str) else outputs)


class EmbeddedPipeline(Pipeline):
    """A normal pipeline whose readers and writers use memory adapters.

    Use send(image) / receive() for monocular inputs, or send({process: image})
    for multiple cameras. All connected reader ports must be supplied; common
    image, timestamp, filename and frame-rate ports are filled automatically.
    receive() returns a dict keyed by the original writer's process.port.
    Processing and sampling stages retain their original configuration.
    """
    def __init__(self, filename, pipeline=None, *, inputs=None, outputs=None):
        self._native = None
        self._started = False
        self._frame = 0
        super().__init__(filename, pipeline=pipeline)
        try:
            _, self._types = _algorithms()
            from kwiver.sprokit.adapters import adapter_data_set, embedded_pipeline
            self._data_set = adapter_data_set.AdapterDataSet
            description = _prepare(self.path, inputs, outputs)
            self.input_names = tuple(description.input_names)
            self._input_ports = description.input_ports
            self._output_ports = description.output_ports
            self.input_ports = tuple(self._input_ports)
            self.output_ports = tuple(self._output_ports)
            if self._work is None:
                self._work = tempfile.TemporaryDirectory(prefix='viame_embedded_')
            self.path = str(Path(self._work.name) / 'embedded.pipe')
            Path(self.path).write_text(description.pipeline_text)
            self._native = embedded_pipeline.EmbeddedPipeline()
            description.build(self._native)
            self._native.start()
            self._started = True
        except Exception:
            self.close()
            raise

    def run(self, *args, **kwargs):
        raise TypeError('Embedded pipelines use send() and receive(), not file-based run()')

    def send(self, images, *, timestamp=None, frame_rate=1.0, values=None):
        """Send a frame or synchronized camera frames, with optional metadata.

        Native ImageContainers and numpy image arrays are accepted. The default
        timestamp counts from frame 1 at frame_rate Hz (default 1). Override
        individual reader ports using values={'input.file_name': 'frame.png'}.
        Sending may block when queues are full; receive results between sends.
        """
        if self.closed:
            raise ValueError('Pipeline is closed')
        if isinstance(frame_rate, bool) or not math.isfinite(float(frame_rate)) or float(frame_rate) <= 0:
            raise ValueError('frame_rate must be a positive finite number')
        if not isinstance(images, dict):
            if len(self.input_names) != 1:
                raise ValueError('Supply images by input process name: ' + ', '.join(self.input_names))
            images = {self.input_names[0]: images}
        unknown = set(images) - set(self.input_names)
        if unknown:
            raise ValueError('Unknown image inputs: ' + ', '.join(sorted(unknown)))
        values = dict(values or {})
        unknown = set(values) - set(self._input_ports)
        if unknown:
            raise ValueError('Unknown input ports: ' + ', '.join(sorted(unknown)))
        if timestamp is None:
            timestamp = self._types.Timestamp()
            timestamp.set_frame(self._frame + 1)
            timestamp.set_time_seconds(self._frame / float(frame_rate))
        converted = {}
        for name, image in images.items():
            if not isinstance(image, self._types.BaseImageContainer):
                import numpy as np
                if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
                    raise TypeError('Images must be native ImageContainers or 2D/3D numpy arrays')
                image = self._types.ImageContainer(self._types.Image(image))
            converted[name] = image
        from ._io_native import add_double
        data = self._data_set.create()
        for port, alias in self._input_ports.items():
            source, field = port.split('.', 1)
            if port in values:
                value = values[port]
            elif field == 'image' and source in converted:
                value = converted[source]
            elif field == 'timestamp':
                value = timestamp
            elif field in ('file_name', 'image_file_name'):
                value = '{}_{:06d}'.format(source, self._frame + 1)
            elif field == 'frame_rate':
                value = float(frame_rate)
            else:
                raise ValueError('Supply reader port via values: ' + port)
            if field == 'frame_rate':
                add_double(data, alias, float(value))
            else:
                data[alias] = value
        self._native.send(data)
        self._frame += 1

    def receive(self, timeout=None):
        """Wait for a result and return its writer-port values, or None at EOF.

        A pipeline can drop or batch frames. Call receive only when expecting
        output; there is not necessarily one result for every send. Optional
        timeout (seconds) raises TimeoutError without closing the pipeline.
        """
        if self.closed:
            raise ValueError('Pipeline is closed')
        if self._native.at_end():
            return None
        if timeout is not None:
            timeout = float(timeout)
            if not math.isfinite(timeout) or timeout < 0:
                raise ValueError('timeout must be a nonnegative finite number')
            deadline = time.monotonic() + timeout
            while self._native.empty():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('No pipeline output available before timeout')
                time.sleep(min(0.005, remaining))
        data = self._native.receive()
        if data.is_end_of_data():
            return None
        result = {}
        for port, alias in self._output_ports.items():
            try:
                result[port] = data[alias]
            except TypeError:
                # Native frame-rate ports use double, which older adapter
                # bindings do not expose through their generic getter.
                from ._io_native import get_double
                result[port] = get_double(data, alias)
        return result

    def close(self):
        """Finish queued work, discard unread results and release the pipeline."""
        if self.closed:
            return
        try:
            if self._started:
                # Drain concurrently before sending EOF: both adapter queues
                # are bounded, so waiting without draining can deadlock.
                native = self._native
                errors = []
                done = threading.Event()
                def drain():
                    try:
                        while not done.is_set() and not native.at_end():
                            if native.empty():
                                done.wait(0.005)
                            elif native.receive().is_end_of_data():
                                break
                    except Exception as exc:
                        errors.append(exc)
                thread = threading.Thread(target=drain, name='viame-close')
                thread.start()
                try:
                    native.send_end_of_input()
                    native.wait()
                finally:
                    done.set()
                    thread.join()
                if errors:
                    raise errors[0]
        finally:
            self._started = False
            self._native = None
            super().close()
