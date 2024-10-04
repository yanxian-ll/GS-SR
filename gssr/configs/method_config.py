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
Put all the method implementations in one location.
"""

from typing import Dict
import tyro

from gssr.configs.base_config import Config
from gssr.dataloader.colmap_dataloader import ColmapDataLoaderConfig
from gssr.dataloader.pgsr_dataloader import PGSRDataLoaderConfig

from gssr.scene.vanilla_scene import VanillaSceneConfig
from gssr.scene.twodgs_scene import TwoDGSSceneConfig
from gssr.scene.scaffold_scene import ScaffoldSceneConfig
from gssr.scene.octree_scene import OctreeSceneConfig
from gssr.scene.scaffold_2dgs_scene import Scaffold2DGSSceneConfig
from gssr.scene.octree_2dgs_scene import Octree2DGSSceneConfig
from gssr.scene.pgsr_scene import PGSRSceneConfig
from gssr.scene.scaffold_pgsr_scene import ScaffoldPGSRSceneConfig
from gssr.scene.octree_pgsr_scene import OctreePGSRSceneConfig

from gssr.gaussian.vanilla_gaussian import VanillaGaussianConfig
from gssr.gaussian.twod_gaussian import TwoDGaussianConfig
from gssr.gaussian.scaffold_gaussian import ScaffoldGaussianConfig
from gssr.gaussian.octree_gaussian import OctreeGaussianConfig
from gssr.gaussian.pgsr_gaussian import PGSRGaussianConfig


method_configs: Dict[str, Config] = {}
descriptions = {
    "3dgs": "Implementation of 3DGS",
    "scaffold-gs": "Implementation of Scaffold-GS",
    "octree-gs"  : "Implementation of Octree-GS",

    "2dgs": "Implementation of 2DGS",
    "scaffold-2dgs": "Implementation of Scaffold-GS + 2DGS",
    "octree-2dgs"  : "Implementation of Octree-GS + 2DGS",

    "pgsr": "Implementation of PGSR",
    "scaffold-pgsr": "Implementation of Scafflod-GS + PGSR",
    "octree-pgsr"  : "Implementation of Octree-GS + PGSR",
}

method_configs["3dgs"] = Config(
    method_name="3dgs",
    scene=VanillaSceneConfig(
        dataloader=ColmapDataLoaderConfig(
            shuffle=True,
            llffhold=8,
            resolution=-1,
            images="images",
            white_background=False,
        ),
        gaussians=VanillaGaussianConfig(
            max_sh_degree=3,
            percent_dense=0.01,
        ),
        random_background=False,
        lambda_dssim=0.2
    )
)

method_configs["2dgs"] = Config(
    method_name="2dgs",
    scene=TwoDGSSceneConfig(
        dataloader=ColmapDataLoaderConfig(),
        gaussians=TwoDGaussianConfig(),
        depth_ratio = 0.0,
        lambda_normal = 0.05,
        lambda_dist = 0.0,
    )
)

method_configs["scaffold-gs"] = Config(
    method_name="scaffold-gs",
    scene=ScaffoldSceneConfig(
        dataloader=ColmapDataLoaderConfig(),
        gaussians=ScaffoldGaussianConfig(),
        lambda_scaling = 0.01,
    ),
)

method_configs["octree-gs"] = Config(
    method_name="octree-gs",
    scene=OctreeSceneConfig(
        dataloader=ColmapDataLoaderConfig(),
        gaussians=OctreeGaussianConfig(),
    )
)

method_configs["scaffold-2dgs"] = Config(
    method_name="scaffold-2dgs", 
    scene = Scaffold2DGSSceneConfig(
        dataloader=ColmapDataLoaderConfig(),
        gaussians=ScaffoldGaussianConfig(),
    )
)

method_configs["octree-2dgs"] = Config(
    method_name="octree-2dgs",
    scene=Octree2DGSSceneConfig(
        dataloader=ColmapDataLoaderConfig(),
        gaussians=OctreeGaussianConfig(),
    )
)

method_configs["pgsr"] = Config(
    method_name="pgsr",
    scene=PGSRSceneConfig(
        dataloader=PGSRDataLoaderConfig(),
        gaussians=PGSRGaussianConfig(),
    )
)

method_configs["scaffold-pgsr"] = Config(
    method_name="scaffold-pgsr",
    scene=ScaffoldPGSRSceneConfig(
        dataloader=PGSRDataLoaderConfig(),
        gaussians=ScaffoldGaussianConfig(),
    )
)

method_configs["octree-pgsr"] = Config(
    method_name="octree-pgsr",
    scene=OctreePGSRSceneConfig(
        dataloader=PGSRDataLoaderConfig(),
        gaussians=OctreeGaussianConfig(),
    )
)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[  # Don't show unparseable (fixed) arguments in helptext.
    tyro.conf.FlagConversionOff[
        tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
    ]
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
