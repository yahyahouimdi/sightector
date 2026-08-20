# Sightector

**Detect. Protect. Respect.**

Sightector is an integrated online-safety solution that combines a machine-learning harassment detection model with a Chrome browser extension, giving users real-time protection against harmful content and malicious links while they browse.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Browser Extension Setup](#browser-extension-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Model Details](#model-details)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Privacy & Security](#privacy--security)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

---

## Overview

Sightector consists of two integrated components:

| Component | Description |
|---|---|
| **Detection Model Server** | A Flask backend serving a trained Linear SVC model that classifies text for harassment in real time |
| **Browser Extension** | A Chrome extension that lets users scan selected text and page URLs directly from the browser |

All analysis runs against a local server — no user data ever leaves the machine.

## Architecture

```
┌─────────────────────┐        HTTP (JSON)        ┌──────────────────────────┐
│  Chrome Extension    │ ─────────────────────────▶│  Flask Detection Server  │
│  (popup + background)│ ◀───────────────────────── │  (TF-IDF + Linear SVC)  │
└─────────────────────┘                             └──────────────────────────┘
```

The extension captures user-selected text or the active tab's URL, sends it to the local Flask server, and renders the classification result with color-coded, actionable feedback.

## Features

**Backend Server**
- Harassment detection via a trained Linear SVC model with TF-IDF vectorization
- Text normalization pipeline with custom stopword filtering
- Lightweight REST API for programmatic access
- Built-in web form for manual text checks

**Browser Extension**
- Real-time harassment scanning of selected text
- Malicious/suspicious URL detection
- Non-HTTPS site flagging
- Color-coded results with contextual safety guidance

## Technology Stack

| Layer | Technologies |
|---|---|
| Backend | Python 3, Flask, scikit-learn, Pandas, NumPy |
| Frontend / Extension | JavaScript, HTML5, CSS3, Chrome Extensions API |
| Data Science | Jupyter Notebook, Matplotlib, Seaborn, XGBoost (experimental) |
| Model Persistence | Pickle |

## Project Structure

```
sightector/
├── README.md
├── Online Harassment Detecting Models.ipynb   # Model training & experimentation
├── DetectionModelServer/
│   ├── app.py                                 # Flask application entry point
│   ├── requirement.txt                        # Python dependencies
│   ├── dataset.csv                            # Training dataset
│   ├── stopwords.txt                          # Custom stopword list
│   ├── LinearSVCTuned.pkl                      # Trained classifier
│   ├── tfidfvectoizer.pkl                      # Fitted TF-IDF vectorizer
│   └── templates/
│       └── Server.html                        # Manual-check web interface
└── sightector extension/
    ├── manifest.json                          # Extension configuration
    ├── index.html                             # Popup UI
    ├── popup.js                               # Popup logic
    ├── background.js                          # Background service worker
    └── images/                                # Icons and assets
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- Google Chrome or a Chromium-based browser

### Backend Setup

```bash
# 1. Navigate to the server directory
cd DetectionModelServer

# 2. Create and activate a virtual environment
python -m venv venv1
source venv1/bin/activate      # Windows: venv1\Scripts\activate

# 3. Install dependencies
pip install -r requirement.txt

# 4. Start the server
python app.py
```

The server starts at `http://localhost:5000`.

### Browser Extension Setup

1. Open Chrome and go to `chrome://extensions`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked** and select the `sightector extension` folder
4. Confirm the Sightector icon appears in the browser toolbar

## Usage

### Backend Server

- **Programmatic use:** send a `POST` request with JSON text to `http://localhost:5000/` (see [API Reference](#api-reference))
- **Manual use:** open `http://localhost:5000/SecureHerServer` and submit text through the web form

### Browser Extension

1. Highlight text on any webpage
2. Click the Sightector icon to open the popup
3. Click **Scan** to analyze the selected text and view color-coded results
4. Click **Check Link** to assess the safety of the current page's URL

## API Reference

### `POST /`

Analyzes text for harassment.

**Request**
```http
POST http://localhost:5000/
Content-Type: application/json

{
  "text": "String to analyze for harassment"
}
```

**Response — 200 OK**
```json
{
  "harassment": false
}
```

**Response — 400 Bad Request**
```json
{
  "error": "No text provided"
}
```

**Response — 500 Internal Server Error**
```json
{
  "error": "Error description"
}
```

### `GET /SecureHerServer`

Returns an HTML form for manual text submission and analysis.

## Model Details

**Training data:** `dataset.csv`, containing labeled harassment / non-harassment text samples.

**Processing pipeline:**
1. Stopword removal using `stopwords.txt`
2. Lowercase normalization
3. TF-IDF vectorization (`tfidfvectoizer.pkl`)
4. Classification via the tuned Linear SVC model (`LinearSVCTuned.pkl`)

## Configuration

**Custom stopwords** — edit `DetectionModelServer/stopwords.txt` (one word per line) to adjust text filtering.

**Extension permissions** (`manifest.json`):

| Permission | Purpose |
|---|---|
| `tabs` | Access tab information |
| `scripting` | Execute scripts in web pages |
| `storage` | Store user preferences/data |
| `activeTab` | Interact with the active tab |
| `*://localhost/*` | Communicate with the local detection server |

## Troubleshooting

| Issue | Resolution |
|---|---|
| Extension can't connect to server | Confirm the Flask server is running on `http://localhost:5000` and no firewall is blocking localhost |
| Model files not found | Ensure `LinearSVCTuned.pkl` and `tfidfvectoizer.pkl` exist in `DetectionModelServer/` and paths in `app.py` are correct |
| Missing dependencies | Re-run `pip install -r requirement.txt` |
| Stopwords file not found | Confirm `stopwords.txt` exists in `DetectionModelServer/` with one stopword per line |

## Roadmap

- Multi-language harassment detection
- Continued model retraining and accuracy improvements
- Real-time performance monitoring
- In-extension user feedback loop
- Firefox and Edge support
- Advanced analytics dashboard
- Configurable sensitivity thresholds
- Integration with external reporting systems

## Privacy & Security

- All analysis runs locally — no data leaves the user's machine
- The extension communicates exclusively with the local Flask server
- No data is transmitted to third-party or external servers

## Contributing

Contributions are welcome in the following areas:

1. Machine learning model improvements (see the Jupyter notebook)
2. Flask backend enhancements
3. Browser extension UI/UX and functionality
4. Additional language or detection-type support

## License

This project is provided as-is for harassment detection and user-protection purposes.

## Support

1. Check the [Troubleshooting](#troubleshooting) section
2. Review browser console errors (F12)
3. Check Flask server logs for backend issues
4. Verify all configuration files are correctly formatted

---

**Sightector** — Detect, Protect, and Respect online communities.