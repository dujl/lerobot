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
from pathlib import Path

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def test_distributed_write_basic():
    """Test distributed writing with two episodes for different tasks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create repo subdirectory since LeRobotDataset.create will create it
        root = Path(tmpdir) / "test"
        repo_id = "test_distributed-basic"

        # Define features
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": None,
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

        # Now check if we can load the dataset normally
        loaded_dataset = LeRobotDataset(repo_id=repo_id, root=root)

        # Check basic stats
        assert loaded_dataset.num_episodes == 2
        assert loaded_dataset.num_frames == num_frames_ep0 + num_frames_ep1
        assert loaded_dataset.meta.total_episodes == 2
        assert loaded_dataset.meta.total_frames == num_frames_ep0 + num_frames_ep1
        assert len(loaded_dataset.meta.tasks) == 2

        # Check we can access each frame
        for i in range(loaded_dataset.num_frames):
            item = loaded_dataset[i]
            assert "observation.state" in item
            assert "action" in item
            assert item["observation.state"].shape == (6,)
            assert item["action"].shape == (6,)


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

        # Load and verify
        loaded_dataset = LeRobotDataset(repo_id=repo_id, root=root)
        assert loaded_dataset.num_episodes == 3
        assert loaded_dataset.num_frames == 3 * frames_per_episode
        assert len(loaded_dataset.meta.tasks) == 1
        assert "sort the blocks" in loaded_dataset.meta.tasks.index


if __name__ == "__main__":
    test_distributed_write_basic()
    print("\n" + "=" * 50 + "\n")
    test_distributed_write_single_task()
    print("\n✓ All tests passed!")
