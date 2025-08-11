import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import logging
import sys
import argparse
from logging.handlers import TimedRotatingFileHandler

def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    
    file_handler = TimedRotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        when="midnight",
        interval=1,
        backupCount=7,
        encoding='utf-8'
    )
    file_handler.setFormatter(log_formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete.")
    return logger

import threading
import uvicorn
from webapp.app import WebApp
from api.routes import app as api_app

def run_api():
    """Runs the FastAPI server."""
    uvicorn.run(api_app, host="127.0.0.1", port=8000)

def main():
    parser = argparse.ArgumentParser(description="Huadian QA Application")
    parser.add_argument("mode", nargs='?', default="webapp", choices=["api", "webapp"], help="Mode to run the application in (defaults to webapp).")
    args = parser.parse_args()

    setup_logging()

    if args.mode == "api":
        # Run API server only
        run_api()
    elif args.mode == "webapp":
        # Run API server in a background thread and then launch the webapp
        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        
        # Launch the Gradio web app
        webapp = WebApp()
        webapp.launch()

if __name__ == "__main__":
    main()