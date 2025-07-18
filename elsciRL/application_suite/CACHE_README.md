# Import Tool Caching Functionality

This document describes the caching functionality added to the `PullApplications` class in `import_tool.py`.

## Overview

The import tool now automatically caches imported data to improve performance and reduce network requests. When you import applications, the tool:

1. **Checks cache first**: Before downloading from GitHub, it checks if the data is already cached
2. **Saves to cache**: After successful imports, data is saved to a local cache file
3. **Tracks imports**: A log file records all import activities with commit IDs and timestamps
4. **Validates cache**: Cache is validated using commit IDs and source data hashes

## Cache Directory Structure

The caching system creates a directory structure in `.cache`:

```
.cache/
├── import_log.json                    # Import activity log
├── problem1/                          # Problem-specific cache
│   ├── cache_metadata.json           # Cache metadata
│   ├── engine/                       # Engine Python files
│   │   └── sailing.py
│   ├── adapters/                     # Adapter Python files
│   │   ├── adapter1.py
│   │   └── adapter2.py
│   ├── experiment_configs/            # Experiment configuration files
│   │   ├── config1.json
│   │   └── config2.json
│   ├── local_configs/                # Local configuration files
│   │   ├── local_config1.json
│   │   └── local_config2.json
│   ├── prerender_data/               # Prerender data files
│   │   ├── data1.json
│   │   └── data2.json
│   ├── prerender_data_encoded/       # Encoded prerender data (numpy arrays)
│   │   ├── data1.npy
│   │   └── data2.npy
│   ├── prerender_images/             # Image files
│   │   ├── image1.png
│   │   └── image2.jpg
│   └── instructions/                 # Instruction files
│       ├── instruction1.json
│       └── instruction2.json
└── problem2/                          # Another problem's cache
    └── ...
```

## Key Features

### Automatic Caching
```python
from elsciRL.application_suite.import_tool import PullApplications

puller = PullApplications()
result = puller.pull(['sailing'])  # Automatically uses cache if available
```

### Cache Information
```python
# Get information about cached data
cache_info = puller.get_cache_info()
print(cache_info)
```

### Import History
```python
# Get latest import information for a problem
latest_info = puller.get_latest_import_info('sailing')
print(latest_info)
```

### Force Refresh
```python
# Force refresh (ignores cache)
result = puller.force_refresh(['sailing'])
```

### Cache Management
```python
# Clear cache for specific problem
puller.clear_cache('sailing')

# Clear all cache
puller.clear_cache()
```

### Main Branch Status Check
```python
# Check if main branch has been updated
status = puller.check_main_branch_status('sailing')
if status:
    print(f"Needs update: {status['needs_update']}")
    print(f"Current main date: {status['current_main_date']}")
    print(f"Cached main date: {status['cached_main_date']}")
```

### Automatic Main Branch Updates
When importing with `commit_id='main'`, the system automatically:
1. Checks if the main branch has been updated since last cache
2. If updated, pulls fresh data and caches it
3. If unchanged, uses cached data
4. Logs all activities with timestamps and commit IDs

```python
# This will automatically check for updates and pull fresh data if needed
result = puller.pull(['sailing'])  # commit_id='main' in config
```

## Cache Validation

The cache is validated using:
1. **Commit ID**: Ensures the cached data matches the requested commit
2. **Source Hash**: Detects changes in source configuration files
3. **Timestamp**: Records when the data was cached
4. **Main Branch Date Check**: For 'main' branch, checks if the main branch has been updated since last cache

## Log File Structure

The import log (`import_log.json`) contains entries like:
```json
{
  "sailing": [
    {
      "timestamp": "2024-01-15T10:30:00.123456",
      "commit_id": "main",
      "source_hash": "abc123...",
      "cache_hit": false,
      "source_data": {
        "engine_folder": "environments",
        "engine_filename": "sailing.py",
        ...
      }
    }
  ]
}
```

## Cache Metadata

Each cached problem includes metadata:
```python
{
  "cache_metadata": {
    "commit_id": "main",
    "source_hash": "abc123...",
    "timestamp": "2024-01-15T10:30:00.123456",
    "main_branch_date": "2024-01-15T10:30:00Z",  # Only for 'main' branch
    "main_branch_sha": "abc123def456..."          # Only for 'main' branch
  },
  "engine": <engine_class>,
  "adapters": {...},
  "experiment_configs": {...},
  ...
}
```

## Performance Benefits

- **Faster imports**: Cached data loads instantly
- **Reduced network usage**: Avoids re-downloading unchanged data
- **Offline capability**: Can work with previously cached data
- **Version tracking**: Know exactly which version of data you're using
- **Smart main branch updates**: Only re-downloads when main branch has actually changed

## Engine and Adapter File Handling

- **Python files**: Engine and adapter .py files are downloaded and cached as actual Python files
- **Dynamic loading**: When loading from cache, Python files are dynamically imported
- **Path management**: Cache directories are temporarily added to Python path for import
- **Error handling**: Graceful fallback if cached Python files can't be loaded
- **Version consistency**: Ensures cached Python files match the commit version

## Error Handling

The caching system includes robust error handling:
- Graceful fallback if cache files are corrupted
- Automatic cache directory creation
- Detailed logging of cache operations
- Safe cache validation

## Example Usage

See `cache_example.py` for a complete demonstration of the caching functionality.

## File Locations

- Cache directory: `./.cache/`
- Log file: `./.cache/import_log.json`
- Problem cache: `./.cache/problem_name/`
- Engine files: `./.cache/problem_name/engine/`
- Adapter files: `./.cache/problem_name/adapters/`
- Metadata file: `./.cache/problem_name/cache_metadata.json`

The cache directory structure is automatically created when the `PullApplications` class is initialized. 