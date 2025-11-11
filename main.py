import os
import logging
from flask import Flask, render_template, abort, request, Response, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize Flask app
app = Flask(__name__)

# --- CUSTOM LOGGING FORMATTER ---
class CustomFormatter(logging.Formatter):
    def format(self, record):
        # Provide default values for client_ip and user_agent if not present
        record.client_ip = getattr(record, 'client_ip', 'N/A')
        record.user_agent = getattr(record, 'user_agent', 'N/A')
        return super().format(record)

# Set up logging
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter(
    '%(asctime)s %(levelname)s: %(message)s [Path: %(pathname)s:%(lineno)d, IP: %(client_ip)s, UA: %(user_agent)s]'
))
logging.basicConfig(level=logging.INFO, handlers=[handler])

# --- INITIALIZE RATE LIMITER ---
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["150 per day", "30 per hour"],
    storage_uri="memory://"
)

# --- FILTER CODE ---
BLOCKED_PATTERNS = [
    {"type": "startswith", "value": "/wp-admin/"},
    {"type": "startswith", "value": "/wp-includes/"},
    {"type": "startswith", "value": "/wp-content/"},
    {"type": "startswith", "value": "/.git/"},
    {"type": "startswith", "value": "/.env"},
    {"type": "startswith", "value": "/.well-known/"},
    {"type": "startswith", "value": "/admin/"},
    {"type": "in", "value": "wp-signup.php"},
    {"type": "in", "value": "xmlrpc.php"},
    {"type": "in", "value": "wp-login.php"},
    {"type": "in", "value": "wp-conflg.php"},
    {"type": "in", "value": "phpinfo.php"},
    {"type": "in", "value": "config.json"},
    {"type": "in", "value": "lock.php"},
    {"type": "in", "value": "classsmtps.php"},
    {"type": "in", "value": "browse.php"},
    {"type": "in", "value": "403.php"},
    {"type": "in", "value": "doc.php"},
    {"type": "in", "value": "cache-compat.php"},
    {"type": "in", "value": "autoload_classmap.php"},
    {"type": "in", "value": "sodium_compat"},
    {"type": "in", "value": "ID3"},
    {"type": "endswith", "value": ".php"},
]

BLOCKED_USER_AGENTS = [
    "ahrefsbot",
    "semrushbot",
    "mj12bot",
    "dotbot",
    "petalbot",
]

@app.before_request
def block_unwanted_probes():
    path_lower = request.path.lower()
    user_agent = request.headers.get("User-Agent", "Unknown").lower()
    x_forwarded_for_full = request.headers.get('X-Forwarded-For', 'Unknown')
    source_ip = x_forwarded_for_full.split(',')[0].strip() if x_forwarded_for_full != 'Unknown' else request.remote_addr or "Unknown"

    # Log every request
    extra = {'client_ip': source_ip, 'user_agent': user_agent}
    app.logger.info(f"Request: {request.method} {request.path}", extra=extra)

    # Block based on user-agent
    for blocked_ua in BLOCKED_USER_AGENTS:
        if blocked_ua in user_agent:
            app.logger.warning(
                f"Blocked user-agent: Path='{request.path}', User-Agent='{user_agent}', "
                f"IP='{source_ip}', X-Forwarded-For='{x_forwarded_for_full}'"
            )
            return Response("Forbidden: Access is denied.", status=403)

    # Block based on path patterns
    for pattern in BLOCKED_PATTERNS:
        match = False
        pattern_value = pattern["value"]
        if pattern["type"] == "startswith" and path_lower.startswith(pattern_value):
            match = True
        elif pattern["type"] == "endswith" and path_lower.endswith(pattern_value):
            match = True
        elif pattern["type"] == "in" and pattern_value in path_lower:
            match = True
        elif pattern["type"] == "equals" and path_lower == pattern_value:
            match = True
        
        if match:
            app.logger.warning(
                f"Blocked probe: Path='{request.path}', MatchedPatternType='{pattern['type']}', "
                f"MatchedPatternValue='{pattern['value']}', User-Agent='{user_agent}', "
                f"IP='{source_ip}', X-Forwarded-For='{x_forwarded_for_full}'"
            )
            return Response("Forbidden: Access is denied.", status=403)

    return None

# --- ROUTES ---
@app.route('/')
@limiter.limit("50 per hour")
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"Error rendering template 'index.html': {e}", exc_info=True)
        abort(500)

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/safety")
def safety():
    return render_template("safety.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/feedback")
def feedback():
    return render_template("feedback.html")

@app.route('/robots.txt')
def serve_robots():
    try:
        return send_from_directory(app.static_folder, 'robots.txt')
    except Exception as e:
        app.logger.error(f"Error serving robots.txt: {e}", exc_info=True)
        abort(404)

@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception as e:
        app.logger.error(f"Error serving static file '{filename}': {e}", exc_info=True)
        abort(404)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=os.environ.get("FLASK_DEBUG", "False").lower() == "true", host="0.0.0.0", port=port)