# Phase 3: Automatic Synchronization

## Overview

Phase 3 introduces automatic synchronization capabilities that detect catalog changes and incrementally update the Pinecone index. This ensures your RAG system always has the latest product information without manual intervention.

## Key Features

### 1. Change Detection (`src/sync/catalog_monitor.py`)

The `CatalogMonitor` tracks product changes using content-based hashing:

- **Added Products**: New products not in previous catalog
- **Modified Products**: Products with changed content (price, description, etc.)
- **Deleted Products**: Products removed from catalog
- **Field-Level Tracking**: Identifies which specific fields changed

```python
monitor = CatalogMonitor("specialized.com", "data/products.json")
changes = monitor.check_for_changes()

# Returns list of CatalogChange objects:
# - change_type: 'added', 'modified', or 'deleted'
# - product_id: Unique identifier
# - fields_changed: List of modified fields (for updates)
```

### 2. Sync Orchestration (`src/sync/sync_orchestrator.py`)

The `SyncOrchestrator` manages the synchronization process:

```python
orchestrator = SyncOrchestrator(
    brand_domain="specialized.com",
    catalog_path="data/products.json",
    index_name="specialized-hybrid-v2",
    auto_start=True  # Enable automatic monitoring
)

# Automatic triggers:
# - After N changes detected (default: 10)
# - After time threshold (default: 5 minutes)
# - For high-priority changes (deletions, price changes)
```

### 3. CLI Interface (`catalog_sync.py`)

Simple command-line interface for all sync operations:

#### Monitor Changes
```bash
python catalog_sync.py monitor specialized.com data/products.json
```

Output:
```
📊 Monitoring catalog: data/products.json
Brand: specialized.com
------------------------------------------------------------

✅ Detected 3 changes:

ADDED (1):
  - BIKE-9876

MODIFIED (2):
  - BIKE-1234
    Fields: price, description
  - BIKE-5678
    Fields: availability

📈 Catalog Statistics:
  Total products: 1250
  Last check: 2024-01-15T10:30:00
  Last sync: 2024-01-15T09:00:00
  Sync count: 42
```

#### Manual Sync
```bash
python catalog_sync.py sync specialized.com data/products.json \
    --index specialized-hybrid-v2
```

#### Continuous Monitoring
```bash
python catalog_sync.py watch specialized.com data/products.json \
    --index specialized-hybrid-v2 \
    --interval 300  # Check every 5 minutes
```

#### Check Status
```bash
python catalog_sync.py status specialized.com
```

## Sync Strategies

### 1. Incremental Sync (Default)
- Only processes changed products
- Maintains sync state between runs
- Efficient for regular updates
- Preserves unchanged product embeddings

### 2. Full Sync
- Re-processes entire catalog
- Use when index structure changes
- Triggered with `--force` flag
- Ensures complete consistency

### 3. Priority-Based Sync
Certain changes trigger immediate sync:
- Product deletions (avoid showing unavailable items)
- Price changes (critical for accuracy)
- Stock status changes
- New product launches

## Integration Patterns

### 1. File System Watcher
```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CatalogFileHandler(FileSystemEventHandler):
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
    
    def on_modified(self, event):
        if event.src_path.endswith('.json'):
            self.orchestrator.trigger_manual_sync()

# Watch catalog file
observer = Observer()
observer.schedule(
    CatalogFileHandler(orchestrator),
    path='data/',
    recursive=False
)
observer.start()
```

### 2. Scheduled Sync
```python
import schedule

# Schedule regular syncs
schedule.every(30).minutes.do(
    lambda: orchestrator.trigger_manual_sync()
)

# Schedule daily full sync
schedule.every().day.at("02:00").do(
    lambda: orchestrator.sync_changes(force_full_sync=True)
)
```

### 3. Webhook Integration
```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/catalog-updated', methods=['POST'])
def catalog_updated():
    # Trigger sync on catalog update webhook
    orchestrator.trigger_manual_sync()
    return {'status': 'sync_triggered'}, 200
```

## State Management

### Monitor State (`catalog_monitor_state.json`)
```json
{
  "product_hashes": {
    "BIKE-123": "a3f5d8e9b2c1...",
    "BIKE-456": "7b9e2f3a8d4c..."
  },
  "last_check": "2024-01-15T10:30:00",
  "last_sync": "2024-01-15T09:00:00",
  "sync_count": 42
}
```

### Change Log (`catalog_changes.jsonl`)
```json
{"timestamp": "2024-01-15T10:30:00", "change_type": "modified", "product_id": "BIKE-123", "fields_changed": ["price"]}
{"timestamp": "2024-01-15T10:30:00", "change_type": "added", "product_id": "BIKE-789"}
```

### Pinecone Sync State (`pinecone_sync_state.json`)
```json
{
  "product_hashes": {...},
  "last_sync": "2024-01-15T09:00:00",
  "total_products": 1250,
  "version": "1.0"
}
```

## Performance Characteristics

### Change Detection
- Time: O(n) where n = number of products
- ~1-2 seconds for 10,000 products
- Memory: Proportional to catalog size

### Incremental Sync
- Only processes changed products
- Typical sync: 10-50 products in 5-10 seconds
- Minimal API calls to Pinecone

### Full Sync
- Processes entire catalog
- ~5-10 minutes for 10,000 products
- Use sparingly (daily/weekly)

## Configuration

### Sync Thresholds
```python
# In SyncOrchestrator.__init__
self.monitor.change_threshold = 10      # Trigger after N changes
self.monitor.time_threshold = timedelta(minutes=5)  # Or after N minutes
self.check_interval = 300               # Check every 5 minutes
```

### Retry Logic
```python
self.retry_attempts = 3    # Retry failed syncs
self.retry_delay = 60      # Wait 1 minute between retries
```

### Rate Limiting
```python
# In SyncScheduler
self.rate_limit = 10       # Max 10 syncs per hour
```

## Error Handling

### Transient Failures
- Network issues: Automatic retry with exponential backoff
- API rate limits: Respect Pinecone limits, queue changes
- Partial failures: Track and retry failed items

### Data Integrity
- Atomic updates: All-or-nothing sync batches
- Rollback capability: Restore previous state if needed
- Validation: Ensure product data integrity before sync

## Monitoring & Alerts

### Metrics to Track
- Changes detected per check
- Sync success/failure rate
- Average sync duration
- Products synced per hour
- Error frequency and types

### Alert Conditions
- Sync failures > threshold
- No changes detected for extended period
- Sync duration > expected
- High error rate

## Best Practices

1. **Start with Manual Monitoring**
   - Run `monitor` command to understand change patterns
   - Verify changes are detected correctly
   - Test sync with small batches

2. **Gradual Automation**
   - Begin with manual syncs
   - Add scheduled syncs during off-hours
   - Enable continuous monitoring when confident

3. **Regular Full Syncs**
   - Schedule weekly full syncs for consistency
   - Run after major catalog restructuring
   - Use to recover from any drift

4. **Monitor Performance**
   - Track sync duration trends
   - Watch for increasing change volumes
   - Optimize batch sizes if needed

5. **Backup Strategy**
   - Keep catalog snapshots
   - Backup monitor state regularly
   - Document recovery procedures

## Troubleshooting

### Issue: Changes not detected
```bash
# Check monitor state
cat accounts/specialized.com/catalog_monitor_state.json

# Force recheck
python catalog_sync.py monitor specialized.com data/products.json
```

### Issue: Sync failures
```bash
# Check logs for errors
# Verify Pinecone API key
export PINECONE_API_KEY=your-key

# Try manual sync with verbose output
python catalog_sync.py sync specialized.com data/products.json --index your-index --force
```

### Issue: Duplicate products
```bash
# Force full sync to rebuild
python catalog_sync.py sync specialized.com data/products.json --index your-index --force
```

## Next Steps

With Phase 3 complete, the RAG system now has:
- ✅ Universal product processing (Phase 1)
- ✅ Hybrid search capabilities (Phase 2)
- ✅ Automatic synchronization (Phase 3)

The final phase will add:
- Phase 4: System integration with Langfuse
- Advanced monitoring and observability
- Production deployment patterns