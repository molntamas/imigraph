# Headless Chromium for docker/check-pages.py (compose service "verify"). The Playwright image
# ships the browsers; the Python binding of the same version is added here.
FROM mcr.microsoft.com/playwright/python:v1.64.1-noble
RUN pip install --no-cache-dir --break-system-packages playwright==1.64.1
