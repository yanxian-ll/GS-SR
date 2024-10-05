# Copyright 2022 The Nerfstudio Team. All rights reserved.
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

"""
Code to train model.
"""
import os
import functools
from typing import Dict, List, Tuple

import torch
from rich.console import Console
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gssr.configs import base_config as cfg
from gssr.engine.callbacks import TrainingCallback, TrainingCallbackLocation
from gssr.scene.base_scene import Scene
from gssr.utils.tensorboard_utils import *
from gssr.utils.image_utils import psnr

CONSOLE = Console(width=120)

class Trainer:
    scene: Scene
    callbacks: List[TrainingCallback]

    def __init__(self, config: cfg.Config, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = "cpu" if world_size == 0 else f"cuda:{local_rank}"
        
        self._start_step = 0
        self.base_dir = config.get_base_dir()

        # directory to save gaussians
        self.gaussian_dir = config.get_gaussian_dir()
        CONSOLE.log(f"Saving gaussians to: {self.gaussian_dir}")

        # directory to save checkpoints
        self.checkpoint_dir = config.get_checkpoint_dir()
        CONSOLE.log(f"Saving checkpoints to: {self.checkpoint_dir}")

        # set up tensorboard
        self.tb_writer = None
        if config.is_tensorboard_enabled():
            self.log_dir = self.base_dir / config.relative_log_dir
            self.tb_writer = SummaryWriter(self.log_dir)
            CONSOLE.log(f"Saving logs to: {self.log_dir}")
        else:
            CONSOLE.log("Tensorboard not available: not logging progress")

    def setup(self, eval=False):
        """Setup the Trainer by calling other setup functions."""
        self.scene: Scene = self.config.scene.setup(
            source_dir = self.config.source_path, eval = eval,
            device = self.device, world_size = self.world_size, local_rank = self.local_rank,
        )
        self.scene._gaussians.setup_optimizers()

        self._load_gaussians()
        self._load_checkpoint()
        self.callbacks = self.scene.get_training_callbacks()

    def train(self) -> None:
        """Train the model."""
        assert self.scene.dataloader.getTrainData() is not None, "Missing DatsetInputs"

        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)

        num_iterations = self.config.trainer.iterations
        self._start_step += 1
        ema_loss_for_log = 0
        progress_bar = tqdm(range(self._start_step, num_iterations), desc="Training progress")
        for step in range(self._start_step, num_iterations + 1):
            iter_start.record()
            # training callbacks before the training iteration
            for callback in self.callbacks:
                callback.run_callback_at_location(step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)

            # time the forward pass
            model_output, loss_dict, metrics_dict = self.scene.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())
            loss.backward()
            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if step % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if step == num_iterations:
                    progress_bar.close()

                # TODO: Log and save
                if (step in self.config.trainer.test_iterations):
                    self.evaluation(step)

                if (step in self.config.trainer.save_iterations):
                    self.save_gaussians(step)
                write_scalar(self.tb_writer, "loss", loss, step)
                write_scalar_dict(self.tb_writer, "loss", loss_dict, step)

                # densify
                self.scene.densify(step, model_output)
                
                # # training callbacks after the training iteration
                # for callback in self.callbacks:
                #     callback.run_callback_at_location(step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)
                
                # Optimizer step
                if step < num_iterations:
                    self.scene._gaussians.optimizer.step()
                    self.scene._gaussians.optimizer.zero_grad(set_to_none = True)

                # Save checkpoint
                if (step in self.config.trainer.checkpoint_iterations):
                    CONSOLE.log(f"\n[ITER {step}] Saving Checkpoint.")
                    self.save_checkpoint(step)

    @torch.no_grad()
    def evaluation(self, step):
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': self.scene.dataloader.getTestData()},
                              {'name': 'train', 'cameras': self.scene.dataloader.getTrainData()})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(self.scene.eval_render(viewpoint)['render'], 0.0 ,1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(self.device), 0.0 , 1.0)
                    l1_test += self.scene.l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                CONSOLE.log(f"\n[ITER {step}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}.")


    
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers."""
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path = self.checkpoint_dir / f"step-{step:05d}.ckpt"
        torch.save((self.scene._gaussians.capture(), step), ckpt_path)
        # possibly delete old checkpoints
        if self.config.trainer.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.trainer.load_ckpt_dir
        if load_dir is not None:
            load_step = self.config.trainer.load_ckpt_step
            if load_step is None:
                print(f"Loading latest checkpoint from {load_dir}")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path = load_dir / f"step-{load_step:05d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            model_params, self._start_step = torch.load(load_path)
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.scene._gaussians.restore(model_params)
            self.scene._gaussians.setup_optimizers()
            CONSOLE.print(f"done loading checkpoint from {load_path}")
        else:
            CONSOLE.print("No checkpoints to load, training from scratch")


    def save_gaussians(self, step: int) -> None:
        """Save the gaussians to the given path"""
        if not self.gaussian_dir.exists():
            self.gaussian_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.gaussian_dir / f"iteration_{step}.ply"
        self.scene._gaussians.save_gaussians(save_path)
        self.scene._gaussians.save_mlp_checkpoints(self.gaussian_dir)

    def _load_gaussians(self) -> None:
        """Load the gaussians from the given path"""
        load_dir = self.config.trainer.load_gaussian_dir
        if load_dir is not None:
            load_step = self.config.trainer.load_gaussian_step
            if load_step is None:
                print(f"Loading latest gaussians from {load_dir}")
                load_step = max([int(x[x.find("_") + 1 : x.find(".")]) for x in os.listdir(load_dir) if x.endswith('.ply')])
            load_path = load_dir / f"iteration_{load_step}.ply"
            self.scene._gaussians.load_gaussians(load_path)
            self.scene._gaussians.load_mlp_checkpoints(load_dir)
            CONSOLE.print(f"done loading gaussians from {load_path}")
        else:
            CONSOLE.print("No gaussians to load")

