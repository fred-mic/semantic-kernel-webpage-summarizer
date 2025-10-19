# semantic-kernel-webpage-summarizer

A small command-line utility that fetches a web page, extracts the visible text, and generates a concise 3–5 bullet summary using the Semantic Kernel SDK with OpenAI (chat completion) as the backend.

This repository contains a minimal example demonstrating:

- Safe-ish fetching of remote web pages (hostname validation to mitigate SSRF)
- Lightweight HTML text extraction
- Summarization via the Semantic Kernel and OpenAI models

## Features

- CLI prompt to enter a URL
- Hostname resolution checks to avoid fetching internal/private IPs
- HTML-to-text extraction using Python's `html.parser`
- Summarization using Semantic Kernel and the OpenAI chat completion connector

## Requirements

- Python 3.10+
- An OpenAI API key with permission to call the chat completion API
- Recommended: a virtual environment

## Installation

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install the expected dependencies manually:

```bash
pip install semantic-kernel httpx python-dotenv
```

## Usage

1. Create an environment file `.env.local` in the project root and set your OpenAI API key:

```text
OPENAI_API_KEY=sk-...your-key-here...
```

2. Run the script:

```bash
python main.py
```

3. Enter the URL you want to summarize when prompted (for example: `https://fredmichael.com`).

## Security and privacy notes

- The script intentionally validates hostnames to reduce SSRF risk, but you should treat external content as untrusted data. Do not send pages containing sensitive information without review.
- The summarization is performed by a remote AI service — be mindful about sending PII or confidential content to the model provider.
- Rotate and store API keys securely (do not commit them to the repository).
- Consider additional safeguards for production: domain allowlists, stricter rate limits, and token-aware truncation.

## Contributing

Suggestions and fixes welcome. Keep changes minimal and include tests when adding behavior.

## License

MIT
