# Phase 1: Ingestion & Catalog Maintenance

A detailed, step-by-step guide to building and running your Python ingestion pipeline for both **product catalog** and **knowledgebase**. Designed to run locally today and easily port to GCP Cloud Run later.

---

## 1. Project Structure

```
ingestion/
├── configs/
│   └── settings.py            # bucket names, index prefixes, models, timeouts
├── src/
│   ├── models/
│   │   └── product.py         # Product data class definition
│   ├── loaders/
│   │   ├── product_catalog_cache.py  # cache & loader logic for GCP product JSON
│   │   ├── product_manager.py        # orchestration of product ingestion
│   │   ├── storage.py                # GCP bucket I/O utilities
│   │   └── pinecone_setup.py         # Pinecone index initialization & config
│   ├── product_ingestor.py    # full + incremental product sync orchestrator
│   ├── knowledge_ingestor.py  # PDF & HTML document ingestion
│   ├── descriptor.py          # descriptor & sizing generators
│   ├── pinecone_client.py     # dense + sparse upsert helpers
│   └── utils.py               # GCP I/O, logging, retry logic
├── tests/                     # unit + integration tests
└── requirements.txt
```

---

## 2. Configuration

* **GCP buckets**

  * Prod: `liddy-account-documents`
  * Dev:  `liddy-account-documents-dev`
  * Paths: `accounts/<brand_url>/products.json` and `accounts/<brand_url>/knowledge/**`

* **Index naming**

  * Dense:  `<env>--<brand_url>--dense`
  * Sparse: `<env>--<brand_url>--sparse`

* **Embedding models**

  * Dense:  `llama-text-embed-v2` (2 048 dims)
  * Sparse: `pinecone-sparse-english-v0`

* **Hybrid blend**

  * Default `alpha = 0.5` (equal weight), tweak per vertical if needed.

* **Descriptor & Sizing**

  * Call LLM only when field is missing or older than 24 hours.
  * Use provided few-shot prompts (see §4.3 & §4.4).

* **Failure & Retry**

  * Exponential backoff (max 3 retries) on API or I/O errors.
  * Log errors with context (`brand_url`, `product_id`, `doc_path`).

---

## 3. Dependency Setup

```bash
pip install -r requirements.txt
# requirements.txt includes:
#   pinecone-client, google-cloud-storage, openai, pdfminer.six, beautifulsoup4, requests, tenacity
```

* Set env vars:

  ```bash
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcp-key.json
  export PINECONE_API_KEY=…
  export PINECONE_ENVIRONMENT=…
  export ENV=dev       # or prod
  export LLM_API_KEY=…
  ```

---

## 4. Product Catalog Ingestion

### 4.1 Full Sync

1. **List** all `products.json` files under `accounts/` in your GCP bucket.
2. **For each** brand URL:

   * Download and parse JSON → list of `Product` objects.
   * Pass to ingestion pipeline (step 4.2).

### 4.2 Incremental Updates

* Accept an **optional** list of changed `product_id`s.
* If provided, only fetch & process those products.

### 4.3 Descriptor Generation

* **Input**: product fields (`name`, `brand`, `category`, `features`, `sizes`, `colors`).
* **Logic**:

  ```python
  if not product.descriptor or product.descriptor.is_stale():
      prompt = build_descriptor_prompt(product)
      product.descriptor = llm.generate(prompt)
  ```
* **Prompt template**:

  ```txt
  You are a product copywriter for the “cycling” vertical.
  Given:
  {
    "brand": "{brand}",
    "name": "{name}",
    "category": "{category}",
    "price": "{price}",
    "features": {features_list},
    "sizes": {sizes_list},
    "colors": {colors_list}
  }
  Produce a single-sentence summary emphasizing use-case and material.
  ```

### 4.4 Sizing Generation

* **Input**: `product.sizing` raw chart + LLM-generated `descriptor`.
* **Logic**:

  ```python
  if not product.sizing or product.sizing.is_stale():
      prompt = SIZING_PROMPT.format_chart(product.raw_sizing, descriptor=product.descriptor)
      product.sizing = llm.generate(prompt)
  ```
* **Prompt**: (as provided, returning only the JSON `sizing` field).

### 4.5 Vector Upsert (Dense & Sparse)

1. **Initialize** Pinecone client with index names per brand and env.
2. **For each** product variant (SKU):

   * Build metadata:

     ```jsonc
     {
       "brand": brand_url,
       "category": product.category,
       "labels": product.labels,
       "size": variant.size,
       "color": variant.color
     }
     ```
   * **Dense**: upsert vector = `embed_llama(descriptor)` + metadata.
   * **Sparse**: upsert vector = `sparse_embed(product_text)` + metadata.
3. **Batch** upserts (e.g. 100 items/batch), handle errors with retry.

---

## 5. Knowledgebase Ingestion

### 5.1 Supported Formats

* **PDF** (`.pdf`): extract text via `pdfminer.six`.
* **HTML** (`.html`, `.htm`): parse via BeautifulSoup, preserve headings.
* (Future: DOCX, PPTX via Apache Tika)

### 5.2 Chunking Strategy

* **Short docs** (< 2 000 tokens): single chunk.
* **Medium** (≤ 20 pages): one chunk per page.
* **Large** (> 20 pages): sliding-window semantic chunks (\~1 000 tokens, 200-token overlap).

### 5.3 Metadata & Vertical Tagging

* **Embed** per chunk:

  ```jsonc
  {
    "content_type": "brand" | "manual" | "how-to" | …,
    "brand": brand_url,
    "vertical": vertical,       // e.g. cycling, fashion
    "source_path": doc_path,
    "page": page_number
  }
  ```
* **Doc classification** via LLM:

  ```txt
  You are a content classifier for the “cycling” vertical.  
  Given this document excerpt, choose one content_type from ["manual","ethos","history","marketing"].
  ```

### 5.4 Upsert into Pinecone

* **Index**: same dense + sparse pattern, but namespace = `knowledge`.
* **Embedding**: use `llama-text-embed-v2` for dense, `pinecone-sparse-english-v0` for sparse.
* **Batch** with retry and log per document.

---

## 6. Execution & Logging

* **Command-line entry points**:

  ```bash
  python src/product_ingestor.py --full-sync
  python src/product_ingestor.py --incremental --ids 123,456
  python src/knowledge_ingestor.py
  ```
* **Logging**: structured JSON logs (`brand_url`, `product_id`, `status`, `error`).
* **Retries**: use `tenacity` for exponential backoff (max 3 attempts).

---

## 7. Next Steps & Rollout

1. **Local testing** with a small brand bucket.
2. **Containerize** and deploy to GCP Cloud Run.
3. **Add scheduler** (Cloud Scheduler) for nightly full sync + hourly incremental.
4. **Monitor** via GCP logs & alerts on failures.
5. **Phase 2**: build the AI agent’s RAG query layer (next document).
