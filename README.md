# Wiki Corpus Information Retrieval Model

## Overview

This project is an information retrieval system designed to search and retrieve information from a wiki corpus. It includes a backend Python script (`search_backend.py`) that implements the retrieval algorithms, a Flask-based frontend (`search_frontend.py`) for handling search requests, and scripts for creating inverted indexes from the wiki corpus and saving them on GCP buckets (`index creation - python notebook pipeline` and `inverted_index_gcp.py`).

## Dependencies

- Python 3.6+
- Flask
- NLTK
- NumPy
- Pandas
- Scikit-learn
- Gensim
- Google Cloud Storage Python Client

Before running the project, ensure you have installed all required packages. You can install them using pip:

```bash
pip install -r requirements.txt
```

Additionally, you will need to download the NLTK stopwords dataset:

```python
import nltk
nltk.download('stopwords')
```

## Setting Up

1. **Google Cloud Storage**: Update the `bucket_name` in `search_backend.py` to match your GCP bucket name.

2. **Inverted Indexes**: Follow the instructions in the "index creation - python notebook pipeline" to generate and store the inverted indexes in your GCP bucket.

3. **Installing dependencies**: Install all dependencies as listed in requirements.txt

## Running the Server

To start the backend server, run the `search_frontend.py` script. This will start a Flask app that listens for search queries:

```bash
python search_frontend.py
```

By default, the server will run on `localhost:5000`. You can issue search queries by navigating to `http://localhost:5000/search?query=your_search_query`.

## API Endpoints

- `/search`: Accepts a `query` parameter with the search terms and returns a JSON list of up to 100 search results, each a tuple of `(wiki_id, title)`.

## Files Description

- `search_backend.py`: Contains the logic for the information retrieval model, including tokenization, querying, and ranking algorithms.
- `search_frontend.py`: Flask application that provides an HTTP interface to the search backend.
- `inverted_index_gcp.py`: Utilities for creating, loading, and manipulating inverted indexes stored in GCP buckets.
- Index Creation Notebook: A Jupyter notebook used for creating inverted indexes from the wiki corpus.



