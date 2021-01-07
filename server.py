from flask import Flask, request
from wget import download
from uuid import uuid1 as uuid
import os

app = Flask(__name__)

APP_VERSION = "1.0"

@app.route('/')
def landing_default():
    return f"StyleGAN Encoder API Endpoint (ver. {APP_VERSION})"

@app.route('/generate/imgurls', methods=['POST'])
def generate_by_imageurls():
    data = request.get_json()

    # Validate the urls field -> TODO: Add specific validation for image urls
    if "urls" not in data:
        return "Request is missing 'urls'", 400

    # Generate job UUID
    job_uuid = uuid()
    job_dir = f"images/{job_uuid}/"

    os.mkdir(job_dir)

    for url in data["urls"]:
        download(url, job_dir)

    return "Success!"
