"""
mjpeg_server.py
HTTP MJPEG streaming server.
Serves live JPEG frames to browser clients.
"""
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler


# Shared state — updated by detection loop, read by HTTP handler
_latest_jpeg = None
_lock = threading.Lock()

DASHBOARD_HTML = b'''<!DOCTYPE html><html><head><title>YOLO ZCU104</title>
<style>body{margin:0;background:#111;color:#0f0;font-family:monospace}
h2{padding:10px;margin:0}img{width:100%;max-width:1280px;display:block}
.i{padding:5px 10px;color:#aaa;font-size:13px}</style></head><body>
<h2>YOLO Object Detection &mdash; ZCU104 FPGA (Real-time)</h2>
<div class="i">Classes: 2-wheelers | auto | bus | car | pedestrian | truck</div>
<img src="/stream"></body></html>'''


def update_frame(jpeg_bytes):
    """Called by detection loop to push new frame."""
    global _latest_jpeg
    with _lock:
        _latest_jpeg = jpeg_bytes


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML)

        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            try:
                while True:
                    with _lock:
                        data = _latest_jpeg
                    if data is None:
                        time.sleep(0.02)
                        continue
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(data)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.033)  # ~30 FPS cap
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # Suppress access logs


def start_server(port=8080):
    """Start HTTP server in a daemon thread. Non-blocking."""
    def _serve():
        HTTPServer(('0.0.0.0', port), _Handler).serve_forever()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return t