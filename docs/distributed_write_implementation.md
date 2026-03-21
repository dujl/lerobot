# Distributed Write Support for LeRobotDataset v3.0

## Overview
This implementation adds distributed/parallel write support for LeRobotDataset v3.0 format, enabling multiple processes/workers to write episodes concurrently.

## Key Components

### 1. LeRobotDatasetMetadata Changes
- Add `load_data` parameter to skip loading actual data when creating dataset for writing
- Initialize empty metadata structures when `load_data=False`

### 2. LeRobotDataset Changes
- Add `load_data` parameter to `__init__` to skip data loading
- Add `save_episode_data()` method to save episode without updating global metadata
- Add `consolidate_episodes()` method to aggregate metadata from multiple workers

## Implementation Details

### save_episode_data()
- Saves parquet and videos from episode_buffer
- Does NOT update info.json, episodes.jsonl, tasks.jsonl
- Returns episode metadata for later consolidation

### consolidate_episodes()
- Aggregates metadata from all saved episodes
- Writes metadata files (info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl)
- Computes and writes aggregated stats.json

## Usage Pattern
```python
# Main process creates empty dataset
main_dataset = LeRobotDataset.create(repo_id="test", fps=30, features=features, root=root)

# Worker 1 writes episode 0
worker1 = LeRobotDataset(repo_id="test", root=root, episodes=[0], load_data=False)
for frame in ep0_frames:
    worker1.add_frame(frame, task="demo")
meta1 = worker1.save_episode_data(episode_index=0, task_index=0, start_index=0, encode_videos=True)

# Worker 2 writes episode 1 (in parallel)
worker2 = LeRobotDataset(repo_id="test", root=root, episodes=[1], load_data=False)
for frame in ep1_frames:
    worker2.add_frame(frame, task="demo")
meta2 = worker2.save_episode_data(episode_index=1, task_index=0, start_index=len(ep0_frames), encode_videos=True)

# Main process consolidates
main_dataset.consolidate_episodes([meta1, meta2])

# Now dataset is complete and can be loaded normally
dataset = LeRobotDataset(repo_id="test", root=root)
