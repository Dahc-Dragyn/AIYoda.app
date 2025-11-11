import requests
import pytest
import os

# Get your Cloud Run service URL from an environment variable
# It's best practice not to hardcode it in tests
BASE_URL = os.environ.get("AIYODA_APP_URL", "https://aiyoda.app")

# --- Tests for robots.txt ---
def test_robots_txt_status_code():
    """Verify robots.txt returns a 200 OK."""
    url = f"{BASE_URL}/robots.txt"
    response = requests.get(url)
    assert response.status_code == 200, f"Expected 200 for {url}, got {response.status_code}"

def test_robots_txt_content():
    """Verify key directives are present in robots.txt content."""
    url = f"{BASE_URL}/robots.txt"
    response = requests.get(url)
    content = response.text
    assert "User-agent: *" in content
    assert "Disallow: /wp-admin/" in content
    assert "Allow: /" in content
    assert "Sitemap: https://aiyoda.app/sitemap.xml" in content
    # You can add more specific content checks if needed

# --- Tests for BLOCKED_PATTERNS ---
@pytest.mark.parametrize("path", [
    "/wp-admin/index.php",
    "/wp-includes/css/style.css",
    "/wp-content/uploads/image.jpg",
    "/.git/HEAD",
    "/.env",
    "/.well-known/security.txt",
    "/admin/dashboard",
    "/wp-signup.php",
    "/xmlrpc.php",
    "/wp-login.php",
    "/some/path/wp-conflg.php", # Note the typo in your BLOCKED_PATTERNS: conflg -> config
    "/phpinfo.php",
    "/api/config.json",
    "/data/lock.php",
    "/includes/classsmtps.php",
    "/browse.php?id=1",
    "/error/403.php",
    "/docs/doc.php",
    "/cache-compat.php",
    "/vendor/autoload_classmap.php",
    "/lib/sodium_compat/test.php",
    "/media/ID3/tag.mp3",
    "/anyfile.php", # Test the .php endswith rule
    "/index.php", # Test the .php endswith rule
])
def test_blocked_paths(path):
    """Verify that blocked paths return a 403 Forbidden."""
    url = f"{BASE_URL}{path}"
    response = requests.get(url)
    assert response.status_code == 403, \
        f"Expected 403 for blocked path '{path}', got {response.status_code}"
    assert "Forbidden: Access is denied." in response.text

# --- Tests for BLOCKED_USER_AGENTS ---
@pytest.mark.parametrize("user_agent", [
    "Mozilla/5.0 (compatible; AhrefsBot/7.0; +http://ahrefs.com/robot/)",
    "SemrushBot/7~bl; +http://www.semrush.com/bot.html)",
    "MJ12bot/v1.4.8 (http://www.majestic12.co.uk/bot.php?+)",
    "DotBot/1.1 (https://www.opensiteexplorer.org/bot)",
    "PetalBot (something.com)",
    "MyCoolBrowser AhrefsBot" # Test partial match
])
def test_blocked_user_agents(user_agent):
    """Verify that requests from blocked user agents return a 403 Forbidden."""
    url = f"{BASE_URL}/" # Test against a valid path, as UA blocking happens before path
    headers = {"User-Agent": user_agent}
    response = requests.get(url, headers=headers)
    assert response.status_code == 403, \
        f"Expected 403 for blocked User-Agent '{user_agent}', got {response.status_code}"
    assert "Forbidden: Access is denied." in response.text

# --- Test for unblocked user agents and paths ---
def test_allowed_access():
    """Verify that a normal request to a valid path is not blocked."""
    url = f"{BASE_URL}/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
    response = requests.get(url, headers=headers)
    assert response.status_code == 200, \
        f"Expected 200 for allowed access, got {response.status_code}"

def test_static_file_access():
    """Verify that access to a known static file (e.g., style.css) is allowed."""
    url = f"{BASE_URL}/static/style.css"
    response = requests.get(url)
    assert response.status_code == 200, \
        f"Expected 200 for static file, got {response.status_code}"
    assert "body" in response.text # Check for some content