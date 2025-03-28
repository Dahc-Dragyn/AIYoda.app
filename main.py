import os
import logging
from flask import Flask, render_template, abort

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error rendering template: {e}")
        abort(500)  # Return HTTP 500 in case of error

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/safety")
def safety():
    return render_template("safety.html") 

@app.route("/terms")
def terms():
    return render_template("terms.html")  

if __name__ == '__main__':
    # Read port from environment or default to 8080
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)