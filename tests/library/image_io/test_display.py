"""The display subprocess can be used from a pipeline worker thread."""
import os
import threading
import numpy as np
import pytest


def test_display_lifecycle():
    if not os.environ.get('DISPLAY') and os.name != 'nt':
        pytest.skip('a display server is required')
    pytest.importorskip('tkinter')
    from viame.image_io import display
    errors = []
    def worker():
        try:
            for value in (0,127,255):
                display.show(np.full((24,32,3),value,np.uint8),delay_ms=20)
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=15)
    assert not thread.is_alive()
    display.close()
    assert not errors
    assert display._process is None
