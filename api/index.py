from http.server import BaseHTTPRequestHandler


STREAMLIT_DEMO_URL = "https://commit-chill-ljf9dpisbrdnw2rc9wbrma.streamlit.app/"


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(307)
        self.send_header("Location", STREAMLIT_DEMO_URL)
        self.end_headers()
