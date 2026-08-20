# Installation Guide

This guide walks through setting up both components of Sightector: the detection model server and the Chrome browser extension.

> 📖 For a project overview, features, and API reference, see [README.md](./README.md).

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Backend Setup](#backend-setup)
- [Browser Extension Setup](#browser-extension-setup)
- [Verifying Your Installation](#verifying-your-installation)

---

## Prerequisites

- Python 3.8+
- pip
- Google Chrome or a Chromium-based browser

## Backend Setup

1. **Navigate to the server directory**
   ```bash
   cd DetectionModelServer
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv1
   source venv1/bin/activate      # Windows: venv1\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

4. **Start the server**
   ```bash
   python app.py
   ```

   The server starts at `http://localhost:5000`.

## Browser Extension Setup

1. Open Chrome and navigate to `chrome://extensions`
2. Enable **Developer mode** using the toggle in the top-right corner
3. Click **Load unpacked**
4. Select the `sightector extension` folder from the project directory
5. Confirm the Sightector icon appears in the browser toolbar

## Verifying Your Installation

1. With the Flask server running, open `http://localhost:5000/SecureHerServer` in your browser — you should see the manual-check form
2. Click the Sightector extension icon — the popup should open without errors
3. Highlight text on any webpage, click the extension icon, and click **Scan** — you should receive a color-coded result

If any step fails, see the [Troubleshooting](./README.md#troubleshooting) section in the README.