#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test for distributed writing functionality in LeRobotDataset v3.0"""

import tempfile
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import load_info, load_stats, load_tasks


def test_distributed_write_basic():
    """Test basic distributed write functionality with two episodes written separately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create repo subdirectory since LeRobotDataset.create will create it
        root = Path(tmpdir) / "test"
        repo_id = "test/distributed-dataset"

        # Create features with just state and action, no videos for faster testing
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["joint"],
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["joint"],
            },
        }

        # Main process creates empty dataset
        main_dataset = LeRobotDataset.create(
            repo_id=repo_id, fps=10, features=features, root=root, use_videos=False
        )

        # Simulate first worker writing episode 0
        worker1 = LeRobotDataset(repo_id=repo_id, root=root, load_data=False)
        num_frames_ep0 = 10
        start_idx = 0
        for i in range(num_frames_ep0):
            frame = {
                "observation.state": np.random.randn(6).astype(np.float32),
                "action": np.random.randn(6).astype(np.float32),
                "task": "pick up the cube",
            }
            worker1.add_frame(frame)

        meta1 = worker1.save_episode_data_and_video(
            chunk_index=0, file_index=0, episode_index=0, task="pick up the cube", task_index=0, global_frame_index=start_idx, encode_videos=False
        )

        # Simulate second worker writing episode 1
        worker2 = LeRobotDataset(repo_id=repo_id, root=root, load_data=False)
        num_frames_ep1 = 15
        start_idx = num_frames_ep0
        for i in range(num_frames_ep1):
            frame = {
                "observation.state": np.random.randn(6).astype(np.float32),
                "action": np.random.randn(6).astype(np.float32),
                "task": "place the cube",
            }
            worker2.add_frame(frame)

        meta2 = worker2.save_episode_data_and_video(
            chunk_index=0, file_index=1, episode_index=1, task="place the cube", task_index=1, global_frame_index=start_idx, encode_videos=False
        )

        # Finalize all workers first - this closes the parquet writers by writing the footer
        worker1.finalize()
        worker2.finalize()

        # Main process consolidates the episodes
        main_dataset.commit_metadata([meta1, meta2])

        # Finalize main dataset
        main_dataset.finalize()

        # Check that all files exist
        assert (root / "meta" / "info.json").exists()
        assert (root / "meta" / "stats.json").exists()
        assert (root / "meta" / "tasks.parquet").exists()

        # Check info.json has correct values
        info = load_info(root)
        assert info["total_episodes"] == 2
        assert info["total_frames"] == num_frames_ep0 + num_frames_ep1
        assert info["total_tasks"] == 2

        # Check tasks has two tasks
        tasks_df = load_tasks(root)
        assert len(tasks_df) == 2
        assert "pick up the cube" in tasks_df.index
        assert "place the cube" in tasks_df.index

        # Check stats exists and has entries for both modalities
        stats = load_stats(root)
        assert "observation.state" in stats
        assert "action" in stats
        assert stats["observation.state"]["count"][0] == num_frames_ep0 + num_frames_ep1

        # Check parquet files exist - depending on size, may be in 1 or 2 files. At least one should exist
        parquet_dir = root / "data" / "chunk-000"
        parquet_files = list(parquet_dir.glob("*.parquet"))
        assert len(parquet_files) >= 1
        assert parquet_dir.exists()

        # Try reading all existing parquet files to confirm they're valid
        import pyarrow.parquet as pq

        total_frames = 0
        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file)
            total_frames += len(table)

        assert total_frames == num_frames_ep0 + num_frames_ep1


def test_distributed_write_single_task():
    """Test distributed writing with multiple episodes all for the same task."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create repo subdirectory since LeRobotDataset.create will create it
        root = Path(tmpdir) / "test"
        repo_id = "test/distributed-single-task"

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["joint"],
            },
            "action": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["joint"],
            },
        }

        main_dataset = LeRobotDataset.create(
            repo_id=repo_id, fps=30, features=features, root=root, use_videos=False
        )

        # Write 3 episodes on different workers
        num_episodes = 3
        frames_per_episode = 5
        current_start = 0
        episode_metas = []
        task_name = "sort the blocks"

        for ep_idx in range(num_episodes):
            worker = LeRobotDataset(repo_id=repo_id, root=root, load_data=False)
            for _ in range(frames_per_episode):
                frame = {
                    "observation.state": np.random.randn(3).astype(np.float32),
                    "action": np.random.randn(3).astype(np.float32),
                    "task": task_name,
                }
                worker.add_frame(frame)

            meta = worker.save_episode_data_and_video(
                chunk_index=0,
                file_index=ep_idx,
                episode_index=ep_idx,
                task=task_name,
                task_index=0,
                global_frame_index=current_start,
                encode_videos=False,
            )
            episode_metas.append(meta)
            current_start += frames_per_episode
            worker.finalize()

        # Consolidate
        main_dataset.commit_metadata(episode_metas)
        main_dataset.finalize()

        # Check that all files exist
        assert (root / "meta" / "info.json").exists()
        assert (root / "meta" / "stats.json").exists()
        assert (root / "meta" / "tasks.parquet").exists()

        # Check info.json has correct values
        info = load_info(root)
        assert info["total_episodes"] == 3
        assert info["total_frames"] == 3 * frames_per_episode
        assert info["total_tasks"] == 1

        # Check tasks has one task
        tasks_df = load_tasks(root)
        assert len(tasks_df) == 1
        assert "sort the blocks" in tasks_df.index

        # Check stats exists
        stats = load_stats(root)
        assert "observation.state" in stats
        assert stats["observation.state"]["count"][0] == 3 * frames_per_episode

        # Check all 3 episodes have their parquet files
        # Since we have small files, all fit in chunk-000 starting with file-000
        expected_files = [f"file-00{i}.parquet" for i in range(num_episodes)]
        found_files = [f.name for f in (root / "data" / "chunk-000").iterdir() if f.suffix == ".parquet"]
        for expected_file in expected_files:
            assert expected_file in found_files


if __name__ == "__main__":
    test_distributed_write_basic()
    print("\n" + "=" * 50 + "\n")
    test_distributed_write_single_task()
    print("\n✓ All tests passed!")
