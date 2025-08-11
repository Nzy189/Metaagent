"""
Web utilities for HTML rendering and browser automation.
Migrated from sweet_rl.sweet_rl.utils.webpage_utils
"""

import os
import re
import time
import traceback
import shutil
import subprocess
import socket
import threading
import http.server
import socketserver
import random
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from datasets import load_dataset as hf_load_dataset
from selenium import webdriver
from selenium.webdriver import FirefoxOptions, ChromeOptions
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.chrome.service import Service as ChromeService


# HTTP server management
_http_server = None
_http_server_thread = None
_server_port = 1234
_server_directory = None


def is_port_available(port):
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False


def is_server_running(port):
    """Check if an HTTP server is running on the given port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        result = s.connect_ex(('localhost', port))
        return result == 0


def load_dataset(num_samples: int = 100, version: str = "v0.2") -> List[Dict]:
        """
        Load samples from WebSight dataset using streaming to avoid downloading the full dataset.
        
        Args:
            num_samples: Number of samples to load
            version: Dataset version
            
        Returns:
            List of formatted samples
        """
        print(f"🔄 Loading {num_samples} samples from HuggingFaceM4/WebSight using streaming...")
        
        try:
            # Use streaming=True to avoid downloading the entire dataset
            ds = hf_load_dataset("HuggingFaceM4/WebSight", version, streaming=True)["train"]
            
            samples = []
            # Use take() to limit the number of samples processed
            for i, example in enumerate(tqdm(ds.take(num_samples), total=num_samples, desc="Loading samples")):
                sample = {
                    "task_id": f"websight_{i}",
                    "problem_description": example["llm_generated_idea"],
                    "ground_truth": replace_urls(example["text"]),
                    "original_index": i
                }
                samples.append(sample)
                
            print(f"✅ Successfully loaded {len(samples)} samples")
            return samples
            
        except Exception as e:
            print(f"❌ Error loading WebSight dataset: {e}")
            print("💡 Trying fallback method without streaming...")
            try:
                # Fallback: load traditional way but limit samples
                ds = hf_load_dataset("HuggingFaceM4/WebSight", version)["train"]
                samples = []
                for i in tqdm(range(min(num_samples, len(ds))), desc="Loading samples"):
                    sample = {
                        "task_id": f"websight_{i}",
                        "problem_description": ds[i]["llm_generated_idea"],
                        "ground_truth": replace_urls(ds[i]["text"]),
                        "original_index": i
                    }
                    samples.append(sample)
                print(f"✅ Successfully loaded {len(samples)} samples using fallback method")
                return samples
            except Exception as e2:
                print(f"❌ Fallback method also failed: {e2}")
                return []


def start_http_server(directory, port=None):
    """Start an HTTP server, randomly picking a free port in range 1000-8000."""
    global _http_server, _http_server_thread, _server_port, _server_directory
    
    # If no port specified, randomly select an available port
    if port is None:
        for _ in range(100):  # Try at most 100 times
            port = random.randint(1000, 8000)
            if is_port_available(port):
                break
        else:
            print("❌ Failed to find an available port")
            return None
    
    # If the server is already running and serving the correct directory, return directly
    if is_server_running(port) and _server_directory == directory:
        print(f"✅ HTTP server already running on port {port}, directory: {directory}")
        return port
    
    # Stop existing server if any
    stop_http_server()
    
    try:
        # Save current working directory
        original_cwd = os.getcwd()
        
        # Create custom request handler with the desired root directory
        class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)
        
        httpd = socketserver.TCPServer(("", port), CustomHTTPRequestHandler)
        
        def run_server():
            print(f"🌐 HTTP server started on port {port}, directory: {directory}")
            httpd.serve_forever()
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        
        # Wait for server to start
        time.sleep(1)
        
        _http_server = httpd
        _http_server_thread = thread
        _server_port = port
        _server_directory = directory
        
        print(f"✅ HTTP server started: http://localhost:{port}")
        return port
        
    except Exception as e:
        print(f"❌ Failed to start HTTP server: {e}")
        return None


def stop_http_server():
    """Stop the HTTP server."""
    global _http_server, _http_server_thread, _server_directory
    
    if _http_server:
        try:
            _http_server.shutdown()
            _http_server.server_close()
            print("🛑 HTTP server stopped")
        except Exception as e:
            print(f"⚠️ Error while stopping HTTP server: {e}")
        finally:
            _http_server = None
            _http_server_thread = None
            _server_directory = None


def replace_urls(text):
    """Replace Unsplash URLs with Picsum URLs for consistent image rendering."""
    # Regular expression to find the URLs
    pattern = r"https://source\.unsplash\.com/random/(\d+)x(\d+)/\?[\w=]+"

    # Function to replace each match with the new URL format
    def replace_match(match):
        width, height = match.groups()
        return f"https://picsum.photos/id/48/{width}/{height}"

    # Use re.sub to replace all occurrences in the text
    new_text = re.sub(pattern, replace_match, text)

    # Make sure that the new text has id 48 for all images
    # Define the regex pattern to match the URLs
    pattern = r"https://picsum\.photos/(\d+)/(\d+)"

    # Define the replacement pattern
    replacement = r"https://picsum.photos/id/48/\1/\2"

    # Use re.sub to replace all matches in the paragraph
    new_text = re.sub(pattern, replacement, new_text)

    return new_text


def check_browser_and_driver_installed(browser_name):
    """Check if both browser and webdriver are properly installed."""
    browser_commands = {
        'chrome': ['google-chrome', 'chrome', 'chromium', 'chromium-browser'],
        'firefox': ['firefox', 'firefox-esr']
    }
    
    driver_commands = {
        'chrome': ['chromedriver'],
        'firefox': ['geckodriver']
    }
    
    # Check if browser exists
    browser_found = False
    for cmd in browser_commands.get(browser_name, []):
        if shutil.which(cmd) is not None:
            browser_found = True
            break
    
    # Check if driver exists
    driver_found = False
    for cmd in driver_commands.get(browser_name, []):
        if shutil.which(cmd) is not None:
            driver_found = True
            break
    
    return browser_found, driver_found


def get_chrome_driver():
    """Try to get Chrome WebDriver with enhanced error handling."""
    try:
        options = ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-background-timer-throttling")
        options.add_argument("--disable-backgrounding-occluded-windows")
        options.add_argument("--disable-renderer-backgrounding")
        
        # Try to find Chrome binary explicitly
        chrome_paths = [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
            "/snap/bin/chromium",
            "/opt/google/chrome/chrome"
        ]
        
        chrome_binary = None
        for path in chrome_paths:
            if os.path.exists(path):
                chrome_binary = path
                print(f"🔍 Found Chrome binary: {path}")
                break
        
        if chrome_binary:
            options.binary_location = chrome_binary
        
        driver = webdriver.Chrome(options=options)
        print("✅ Chrome WebDriver initialized successfully")
        return driver
    except Exception as e:
        print(f"❌ Chrome WebDriver failed: {type(e).__name__}: {e}")
        if "chromedriver" in str(e).lower():
            print("   💡 ChromeDriver not found. Install with: sudo apt-get install chromium-chromedriver")
        return None


def get_firefox_driver():
    """Try to get Firefox WebDriver with enhanced error handling."""
    try:
        options = FirefoxOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        # Try to find Firefox binary explicitly
        firefox_paths = [
            "/usr/bin/firefox",
            "/usr/bin/firefox-esr", 
            "/snap/bin/firefox",
            "/opt/firefox/firefox",
            "/usr/local/bin/firefox"
        ]
        
        firefox_binary = None
        for path in firefox_paths:
            if os.path.exists(path):
                firefox_binary = path
                print(f"🔍 Found Firefox binary: {path}")
                break
        
        if firefox_binary:
            options.binary_location = firefox_binary
        
        driver = webdriver.Firefox(options=options)
        print("✅ Firefox WebDriver initialized successfully")
        return driver
    except Exception as e:
        print(f"❌ Firefox WebDriver failed: {type(e).__name__}: {e}")
        if "binary is not a Firefox executable" in str(e):
            print("   💡 Firefox binary path issue detected")
        return None


def get_driver():
    """
    Get WebDriver instance, trying Chrome first, then Firefox.
    Returns None if no working WebDriver available (for mock mode).
    """
    print("🔄 Initializing WebDriver...")
    
    # Check what browsers and drivers are available
    chrome_browser, chrome_driver = check_browser_and_driver_installed('chrome')
    firefox_browser, firefox_driver = check_browser_and_driver_installed('firefox')
    
    print(f"🔍 Browser and WebDriver availability check:")
    print(f"   Chrome: Browser {'✅' if chrome_browser else '❌'}, Driver {'✅' if chrome_driver else '❌'}")
    print(f"   Firefox: Browser {'✅' if firefox_browser else '❌'}, Driver {'✅' if firefox_driver else '❌'}")
    
    # Try Chrome first (generally more reliable for headless rendering)
    if chrome_browser and chrome_driver:
        print("🔄 Attempting Chrome WebDriver...")
        driver = get_chrome_driver()
        if driver:
            return driver
    
    # Try Firefox as fallback
    if firefox_browser and firefox_driver:
        print("🔄 Attempting Firefox WebDriver...")
        driver = get_firefox_driver()
        if driver:
            return driver
    
    # If we get here, no browsers worked - return None for mock mode
    print("❌ No working WebDriver found, will use mock mode")
    print("   🤖 HTML files will be saved but no actual rendering")
    print("   📁 Results will still be generated for evaluation")
    print("")
    print("💡 To enable real HTML rendering, try these commands:")
    print("")
    
    if not chrome_browser and not firefox_browser:
        print("   📦 Install browsers:")
        print("      sudo apt-get update")
        print("      sudo apt-get install chromium-browser firefox")
        print("")
    
    if not chrome_driver and not firefox_driver:
        print("   🚗 Install WebDrivers:")
        print("      sudo apt-get install chromium-chromedriver")
        print("      # OR for Firefox:")
        print("      sudo apt-get install firefox-geckodriver")
        print("      # OR use automatic installer:")
        print("      pip install webdriver-manager")
        print("")
    
    print("   🖥️  For headless systems (servers without display):")
    print("      sudo apt-get install xvfb")
    print("      xvfb-run -a python your_script.py")
    print("")
    print("   🔧 Quick fix for Ubuntu/Debian:")
    print("      sudo apt-get install chromium-browser chromium-chromedriver xvfb")
    print("      xvfb-run -a python your_script.py")
    print("")
    
    return None  # Return None instead of raising exception


def render_full_html(driver, html_snippet, temp_path, env_id=0):
    """
    Render HTML snippet to image using WebDriver via HTTP server
    
    Args:
        driver: WebDriver instance
        html_snippet: HTML code to render
        temp_path: Temporary directory path
        env_id: Environment ID for file naming
        
    Returns:
        Tuple of (image_path, html_path) or (None, None) if failed
    """
    current_time = time.time()

    # Generate file names (just the names, not full paths)
    html_filename = f"{env_id}_{current_time}.html"
    image_filename = f"{env_id}_{current_time}.png"
    
    # Full paths
    html_file_path = os.path.join(temp_path, html_filename)
    image_path = os.path.join(temp_path, image_filename)
    
    print(f"🔄 [render_full_html] Starting render process...")
    print(f"   📝 HTML length: {len(html_snippet)} characters")
    print(f"   📁 HTML file: {html_file_path}")
    print(f"   🖼️ Image file: {image_path}")
    
    # Start HTTP server if needed
    server_port = start_http_server(temp_path)
    if server_port is None:
        print(f"❌ [render_full_html] Failed to start HTTP server")
        return None, None
    
    try:
        os.makedirs(temp_path, exist_ok=True)
        # Save the HTML snippet to a temporary file
        with open(html_file_path, "w", encoding='utf-8') as file:
            file.write(html_snippet)
        print(f"✅ [render_full_html] HTML file written successfully")
        
        if not os.path.exists(html_file_path):
            print(f"❌ [render_full_html] HTML file creation failed: {html_file_path}")
            return None, None
            
        file_size = os.path.getsize(html_file_path)
        print(f"📏 [render_full_html] HTML file size: {file_size} bytes")
        
    except Exception as write_e:
        print(f"❌ [render_full_html] HTML file write failed: {type(write_e).__name__}: {write_e}")
        return None, None

    try:
        # Build HTTP URL (use file name, not full path)
        file_url = f"http://localhost:{server_port}/{html_filename}"
        print(f"🌐 [render_full_html] Loading URL: {file_url}")
        
        driver.get(file_url)
        print(f"✅ [render_full_html] Page loaded successfully")
        
        # Wait a moment for page to fully load
        time.sleep(1)
        
        # Check if driver is responsive
        try:
            page_title = driver.title
            print(f"📄 [render_full_html] Page title: '{page_title}'")
        except Exception as title_e:
            print(f"⚠️ [render_full_html] Cannot get page title: {title_e}")
        
        # Take screenshot
        print(f"📸 [render_full_html] Taking screenshot...")
        driver.save_screenshot(image_path)
        
        # Verify screenshot was created
        if os.path.exists(image_path):
            image_size = os.path.getsize(image_path)
            print(f"✅ [render_full_html] Screenshot created: {image_path} ({image_size} bytes)")
        else:
            print(f"❌ [render_full_html] Screenshot file not created: {image_path}")
            # Remove HTML file since screenshot failed
            try:
                if os.path.exists(html_file_path):
                    os.remove(html_file_path)
            except:
                pass
            return None, None

        # Keep HTML file (don't cleanup) - as requested for result storage
        print(f"💾 [render_full_html] Keeping HTML file: {html_file_path}")
        print(f"🌐 [render_full_html] Accessible at: {file_url}")
        
        # Stop HTTP server
        stop_http_server()
        print(f"🛑 [render_full_html] HTTP服务器已关闭")
            
        return image_path, html_file_path
        
    except Exception as e:
        import traceback
        print(f"❌ [render_full_html] Rendering failed: {type(e).__name__}: {e}")
        print(f"📋 [render_full_html] Full traceback:")
        traceback.print_exc()
        
        # Additional diagnostic info
        try:
            print(f"🔍 [render_full_html] Diagnostic info:")
            print(f"   - Driver status: {driver is not None}")
            if driver:
                try:
                    current_url = driver.current_url
                    print(f"   - Current URL: {current_url}")
                except:
                    print(f"   - Cannot get current URL")
            print(f"   - HTML file exists: {os.path.exists(html_file_path)}")
            print(f"   - Temp path writable: {os.access(temp_path, os.W_OK)}")
        except Exception as diag_e:
            print(f"   - Diagnostic collection failed: {diag_e}")
        
        # Cleanup on error - only remove HTML file if image creation failed
        try:
            if os.path.exists(html_file_path):
                os.remove(html_file_path)
                print(f"🗑️ [render_full_html] Cleaned up HTML file after error")
        except Exception as cleanup_e:
            print(f"⚠️ [render_full_html] Error cleanup failed: {cleanup_e}")
        
        # Ensure HTTP server is closed on error
        stop_http_server()
        print(f"🛑 [render_full_html] HTTP服务器已关闭（出错后清理）")
            
        return None, None


def extract_html_snippet(paragraph):
    """Extract HTML snippet from text paragraph."""
    # Regular expression pattern to match the entire HTML content
    paragraph = replace_urls(paragraph)
    html_pattern = r"<html.*?>.*?</html>"

    # Search for the HTML snippet in the paragraph
    match = re.search(html_pattern, paragraph, re.DOTALL)

    if match:
        return paragraph.replace(match.group(0), "[SEE RENDERED HTML]"), match.group(0)
    else:
        html_pattern = r"<body.*?>.*?</body>"
        match = re.search(html_pattern, paragraph, re.DOTALL)
        if match:
            return paragraph.replace(
                match.group(0), "[SEE RENDERED HTML]"
            ), match.group(0)
        else:
            return paragraph, None 