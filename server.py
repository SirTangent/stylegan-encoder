from flask import Flask, request
from wget import download
from uuid import uuid1 as uuid
import os

from runtime import Runtime
from scraper import scrape_images_webdriver

app = Flask(__name__)

APP_VERSION = "1.0"

@app.route('/')
def landing_default():
    return f"StyleGAN Encoder API Endpoint (ver. {APP_VERSION})"

# TODO: Work In Progress
@app.route('/generate/imgurls', methods=['POST'])
def generate_by_imageurls():
    data = request.get_json()

    # Validate the urls field -> TODO: Add specific validation for image urls
    if "urls" not in data:
        return "Request is missing 'urls'", 400

    # TODO: Use Runtime init method for uuid gen
    # Generate job UUID
    job_uuid = uuid()
    job_dir = f"images/{job_uuid}/"

    os.mkdir(job_dir)

    for url in data["urls"]:
        download(url, job_dir)

    rt = Runtime(job_dir)
    return rt.run_pipeline()

@app.route('/generate/pageurl', methods=['PUT'])
def generate_by_webpage():

    # TODO: Use proper validation library
    if "url" not in request.headers:
        return "Header missing 'url' key.", 400

    # Initialize ML runtime
    runtime = Runtime(generate_folder=True)

    # Run scraping library
    successful, total = scrape_images_webdriver(request.headers['url'], runtime.dir)

    # Stop and report error if applicable
    if successful <= 0:
        return f"No images found on webpage!", 404

    # Run Pipeline
    pipeline_codes = runtime.run_pipeline()

    return str(sum(pipeline_codes))
