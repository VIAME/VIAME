"""RGB image windows using Tk and Pillow, isolated from pipeline worker threads.

Tk is part of standard Python distributions with desktop support. On Linux,
install the distribution's python3-tk package to enable these GUI features.
"""
import atexit
import io
import json
import struct
import subprocess
import sys
import threading

_process = None
_lock = threading.Lock()


def show(image, title="VIAME", delay_ms=0):
    """Show an RGB array; wait for a key, window close, or a positive timeout."""
    from PIL import Image
    global _process
    payload = io.BytesIO()
    Image.fromarray(image).save(payload, format="PNG")
    encoded = payload.getvalue()
    header = json.dumps(dict(title=str(title), delay_ms=int(delay_ms))).encode()
    with _lock:
        if _process is None or _process.poll() is not None:
            _process = subprocess.Popen(
                [sys.executable, __file__], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            _process.stdin.write(struct.pack("!II", len(header), len(encoded)))
            _process.stdin.write(header)
            _process.stdin.write(encoded)
            _process.stdin.flush()
            ready = _process.stdout.read(1)
        except BrokenPipeError:
            ready = b""
        if ready != b"1":
            error = _process.stderr.read().decode(errors="replace")
            _process.wait()
            _process = None
            raise RuntimeError("Cannot display image (Tk and a desktop display are required): " + error)


def close():
    """Release the display process and all its windows."""
    global _process
    with _lock:
        if _process is not None:
            _process.stdin.close()
            try:
                _process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                _process.terminate()
                _process.wait()
            _process.stdout.close()
            _process.stderr.close()
            _process = None


atexit.register(close)


def _serve():
    import queue
    import tkinter as tk
    from PIL import Image, ImageTk
    root = tk.Tk()
    root.withdraw()
    windows = {}
    requests = queue.Queue()
    active = None
    timer = None

    def read_requests():
        stream = sys.stdin.buffer
        try:
            while True:
                sizes = stream.read(8)
                if not sizes:
                    break
                if len(sizes) != 8:
                    raise EOFError("incomplete display request")
                header_size, image_size = struct.unpack("!II", sizes)
                header = json.loads(stream.read(header_size))
                picture = Image.open(io.BytesIO(stream.read(image_size)))
                picture.load()
                requests.put((header, picture))
        except Exception as error:
            requests.put(error)
        finally:
            requests.put(None)

    def finish(title):
        nonlocal active, timer
        if active != title:
            return
        active = None
        if timer is not None:
            root.after_cancel(timer)
            timer = None
        sys.stdout.buffer.write(b"1")
        sys.stdout.buffer.flush()

    def poll():
        nonlocal active, timer
        # Reading stdin on a separate thread keeps windows responsive between
        # frames, including while the pipeline is doing inference.
        try:
            request = requests.get_nowait()
        except queue.Empty:
            root.after(10, poll)
            return
        if request is None or isinstance(request, Exception):
            if isinstance(request, Exception):
                print(str(request), file=sys.stderr)
            root.quit()
            return
        header, picture = request
        title = header["title"]
        if title not in windows or not windows[title][0].winfo_exists():
            window = tk.Toplevel(root)
            window.title(title)
            label = tk.Label(window)
            label.pack(fill="both", expand=True)
            windows[title] = (window, label)
            window.bind("<Key>", lambda event, name=title: finish(name))
            def closed(name=title, widget=window):
                finish(name)
                widget.destroy()
            window.protocol("WM_DELETE_WINDOW", closed)
        window, label = windows[title]
        rendered = ImageTk.PhotoImage(picture, master=root)
        label.configure(image=rendered)
        label.image = rendered
        active = title
        if header["delay_ms"] > 0:
            timer = root.after(header["delay_ms"], lambda: finish(title))
        window.deiconify()
        window.focus_force()
        root.after(10, poll)

    threading.Thread(target=read_requests, daemon=True).start()
    root.after(0, poll)
    try:
        root.mainloop()
    finally:
        root.destroy()


if __name__ == "__main__":
    _serve()
