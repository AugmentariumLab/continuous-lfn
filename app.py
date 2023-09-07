"""This is the main app entrypoint for training and inference."""
import dataclasses
import itertools
import os
import random
import shutil
import subprocess
import time
import json
import copy
from typing import Optional, Union, Tuple, Sequence, Dict, Any
import glob

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2

import options
from data_readers.holostudio_reader import HolostudioDataset
from data_readers.video_reader import VideoReader
from data_readers import holostudio_reader_utils
from data_readers.nerf_reader import NerfSyntheticDataset
from models import adaptive_mlp
from models import adaptive_resnet
from models import atlasnet
from models import learned_ray
from models import mlp
from models import resmlp
from models import multimodel
from models import subnet
from models import video_layers
from models import wrapper_models
from models import multisubnet
from models import multiresnet
from models import multimipnet
from models import subnet_svd
from models import svdnet
from models import mipnet
from models import smipnet
from models import pruning_mlp
from models import slimnet
from models import streamable_nf
from models import lcnet
from models import lcmipnet
from models import lcslimnet
from models import multislnet
from protos_compiled import model_pb2
from utils import my_torch_utils
from utils import my_utils
from utils import nerf_utils
from utils import torch_checkpoint_manager
from utils import pytorch_psnr
from utils import pytorch_ssim
from utils import sat_utils
from utils import fft_flicker
from utils import visualize_lfn_weights
from options import (ADAPTIVE_MLP, MLP, RESMLP, MULTIMIPNET, MULTIMODEL, ADAPTIVE_RESNET, LEARNED_RAY,
                     ATLASNET, SUBNET, SUBNET_SVD, MULTISUBNET, MULTIRESNET,
                     SVDNET, MIPNET, SMIPNET, PRUNINGMLP, SLIMNET, STREAMABLE,
                     LCNET, LCMIPNET, LCSLIMNET, MULTISLNET)

ADAPTIVE_NETWORKS = {ADAPTIVE_RESNET, SUBNET, SUBNET_SVD,
                     MULTISUBNET, MULTIRESNET, MULTISLNET, SVDNET, MIPNET, SMIPNET,
                     SLIMNET, LCNET, MULTIMIPNET, LCMIPNET, LCSLIMNET}


class CheckpointKeys:
    MODEL_STATE_DICT = "model_state_dict"
    LATENT_CODE_STATE_DICT = "latent_code_state_dict"
    OPTIMIZER_STATE_DICT = "optimizer_state_dict"
    LATENT_OPTIMIZER_STATE_DICT = "latent_optimizer_state_dict"
    AUX_MODEL_STATE_DICT = "aux_model_state_dict"
    IMPORTANCE_MODEL_STATE_DICT = "importance_model_state_dict"


@dataclasses.dataclass
class InferenceOutputs:
    model_output: Union[torch.Tensor,
                        adaptive_mlp.AdaptiveMLPOutputs,
                        multimodel.MultiModelMLPOutputs,
                        adaptive_resnet.AdaptiveResnetOutputs,
                        atlasnet.AtlasNetOutputs,
                        subnet.SubNetOutputs,
                        subnet_svd.SubNetSVDOutputs,
                        multisubnet.MultiSubnetOutputs,
                        svdnet.SVDNetOutputs]
    loss: torch.Tensor
    aux_output: Optional[torch.Tensor] = None
    color_loss: Optional[torch.Tensor] = None
    efficiency_loss: Optional[torch.Tensor] = None
    importance_loss: Optional[torch.Tensor] = None
    loadbalance_loss: Optional[torch.Tensor] = None
    clustering_loss: Optional[torch.Tensor] = None
    mm_importance_loss: Optional[torch.Tensor] = None
    lc_loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    ray_colors: Optional[torch.Tensor] = None
    aux_discarding_keep: Optional[torch.Tensor] = None


@dataclasses.dataclass
class RenderFullImageOutputs:
    full_image: torch.Tensor
    aux_image: Optional[torch.Tensor] = None
    expected_layers_image: Optional[torch.Tensor] = None
    layer_outputs: Optional[torch.Tensor] = None
    mm_model_selection: Optional[torch.Tensor] = None
    mm_model_top_selection_indices: Optional[Tuple[torch.Tensor,
                                                   torch.Tensor]] = None
    uvmap_outputs: Optional[torch.Tensor] = None


class App:
    def __init__(self):
        self.args = args = options.parse_options()
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        self.device = my_torch_utils.find_torch_device(self.args.device)
        self.rng = np.random.default_rng(args.random_seed)
        if args.script_mode not in (options.GET_TRAIN_TIME,):
            self.train_data, self.val_data, self.test_data = self.setup_datasets()
            self.model, self.latent_code_layer, self.importance_model, self.aux_model = self.setup_model()
            (self.checkpoint_manager, self.writer, self.optimizer, self.latent_optimizer,
             self.lr_scheduler) = self.setup_checkpoints()

    def start(self):
        """Start the program"""
        args = self.args
        try:
            if args.script_mode == options.TRAIN:
                self.run_training_loop()
            elif args.script_mode == options.INFERENCE:
                self.run_inference()
            elif args.script_mode == options.VISUALIZE_CAMERAS:
                self.visualize_cameras()
            elif args.script_mode == options.SAVE_PROTO:
                self.run_save_proto()
            elif args.script_mode in (options.BENCHMARK, options.BENCHMARK_SLIMNET):
                self.run_benchmark()
            elif args.script_mode == options.VIEWER:
                self.run_viewer()
            elif args.script_mode == options.DEBUG:
                self.run_debug()
            elif args.script_mode in (options.EVAL, options.EVAL_MULTIRES,
                                      options.EVAL_MEMORIZATION,
                                      options.EVAL_MEMORIZATION_MULTIRES,
                                      options.EVAL_SLIMNET_LODS):
                self.run_eval()
            elif args.script_mode == options.GET_TRAIN_TIME:
                self.run_get_train_time()
            elif args.script_mode == options.RENDER_OCCUPANCY_MAP:
                self.run_render_occupancy_map()
            elif args.script_mode == options.RENDER_TRANSITION:
                self.run_render_transition()
            elif args.script_mode == options.RENDER_FOVEATION:
                self.run_render_foveation()
            elif args.script_mode == options.RENDER_SLIMNET_LEVELS:
                self.run_render_slimnet_levels()
            elif args.script_mode == options.RENDER_SLIMNET_TEASER:
                self.run_render_slimnet_teaser()
            elif args.script_mode == options.RENDER_SLIMNET_LOOP:
                self.run_render_slimnet_loop()
            elif args.script_mode in (options.EVAL_FLICKER, options.EVAL_FLICKER_DITHER):
                self.run_eval_flicker()
            elif args.script_mode == options.EVAL_UPDATE_MODELSIZE:
                self.run_eval_update_modelsize()
            elif args.script_mode == options.RENDER_LOD_EPI:
                self.run_render_lod_epi()
            elif args.script_mode == options.VISUALIZE_LFN_WEIGHTS:
                return visualize_lfn_weights.visualize_lfn_weights(self)
            elif args.script_mode == options.RENDER_NEURON_MASKING_EXAMPLE:
                self.run_render_neuron_masking_example()
            elif args.script_mode == options.DUMP_MODEL_SIZES:
                self.dump_model_sizes()
            else:
                raise ValueError("Unknown script mode")
        except KeyboardInterrupt:
            pass
        finally:
            if hasattr(self, "writer") and self.writer is not None:
                self.writer.close()
        print("Done")

    def setup_datasets(self):
        args = self.args
        if args.dataset == options.VIDEO:
            raise NotImplementedError("Not implemented")
        elif args.dataset == options.HOLOSTUDIO:
            train_data = HolostudioDataset(
                args.dataset_path, args.dataset_resize_factor,
                load_saliency=args.dataset_load_saliency,
                max_frames=args.dataset_max_frames,
                load_every_nth_view=args.dataset_loadeverynthview,
                renderposes_height=args.dataset_render_poses_height,
                mip_factors=args.dataset_mip_factors or (),
                resize_interpolation=args.dataset_interp_mode,
                dropout_poses=(args.dataset_ignore_poses or []) +
                (args.dataset_val_poses or []) +
                (args.dataset_test_poses or []),
                renderposes_centeroffset=args.dataset_render_poses_centeroffset,
                renderposes_rotations=args.dataset_render_poses_rotations,
                renderposes_flip_up=args.dataset_render_poses_flip_up,
                cache_lowres=args.cache_lowres,
                saliency_dirname=args.saliency_dirname
            )
            val_data = None
            test_data = None
            if args.dataset_val_poses:
                val_data = HolostudioDataset(
                    args.dataset_path, args.dataset_resize_factor,
                    max_frames=args.dataset_max_frames,
                    renderposes_height=args.dataset_render_poses_height,
                    mip_factors=args.dataset_mip_factors or (),
                    resize_interpolation=args.dataset_interp_mode,
                    dropout_poses=(args.dataset_val_poses or []),
                    include_dropout_only=True,
                    renderposes_centeroffset=args.dataset_render_poses_centeroffset)
            if args.dataset_test_poses:
                test_data = HolostudioDataset(
                    args.dataset_path, args.dataset_resize_factor,
                    max_frames=args.dataset_max_frames,
                    renderposes_height=args.dataset_render_poses_height,
                    mip_factors=args.dataset_mip_factors or (),
                    resize_interpolation=args.dataset_interp_mode,
                    dropout_poses=(args.dataset_test_poses or []),
                    include_dropout_only=True,
                    renderposes_centeroffset=args.dataset_render_poses_centeroffset)
        elif args.dataset == options.NERFSYNTHETIC:
            train_data = NerfSyntheticDataset(
                args.dataset_path, args.dataset_resize_factor,
                max_frames=args.dataset_max_frames,
                renderposes_height=args.dataset_render_poses_height,
                mip_factors=args.dataset_mip_factors or (),
                resize_interpolation=args.dataset_interp_mode,
                dropout_poses=(args.dataset_ignore_poses or []) +
                (args.dataset_val_poses or []) +
                (args.dataset_test_poses or []))
            val_data = None
            test_data = None
            if args.dataset_val_poses:
                val_data = NerfSyntheticDataset(
                    args.dataset_path, args.dataset_resize_factor,
                    max_frames=args.dataset_max_frames,
                    renderposes_height=args.dataset_render_poses_height,
                    mip_factors=args.dataset_mip_factors or (),
                    resize_interpolation=args.dataset_interp_mode,
                    dropout_poses=(args.dataset_val_poses or []),
                    include_dropout_only=True)
            if args.dataset_test_poses:
                test_data = NerfSyntheticDataset(
                    args.dataset_path, args.dataset_resize_factor,
                    max_frames=args.dataset_max_frames,
                    renderposes_height=args.dataset_render_poses_height,
                    mip_factors=args.dataset_mip_factors or (),
                    resize_interpolation=args.dataset_interp_mode,
                    dropout_poses=(args.dataset_test_poses or []),
                    include_dropout_only=True)
        else:
            raise ValueError(f"Unknown dataset {args.dataset}")
        return train_data, val_data, test_data

    def setup_model(self) -> Tuple[nn.Module, nn.Module, Optional[nn.Module], Optional[nn.Module]]:
        args = self.args
        base_feature_size = 6
        latent_code_dim = args.latent_code_dim if args.num_latent_codes >= 0 else 1
        in_features = (base_feature_size + 2 * base_feature_size * args.positional_encoding_functions +
                       2 * args.sh_encoding_degree ** 2 +
                       latent_code_dim)
        out_features = 3 if not args.predict_alpha else 4
        if args.model == ADAPTIVE_MLP:
            model = adaptive_mlp.AdaptiveMLP(in_features=in_features,
                                             out_features=out_features,
                                             layers=args.model_layers,
                                             hidden_features=args.model_width).to(self.device)
        elif args.model == MLP:
            model = mlp.MLP(in_features=in_features,
                            out_features=out_features,
                            layers=args.model_layers,
                            hidden_features=args.model_width,
                            use_layernorm=args.model_use_layernorm).to(self.device)
        elif args.model == RESMLP:
            model = resmlp.ResMLP(in_features=in_features,
                                  out_features=out_features,
                                  layers=args.model_layers,
                                  hidden_features=args.model_width,
                                  use_layernorm=args.model_use_layernorm).to(self.device)
        elif args.model == MULTIMODEL:
            clustering_feature_size = base_feature_size + out_features
            model = multimodel.MultiModelMLP(
                num_models=args.multimodel_num_models,
                in_features=in_features,
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                use_layernorm=args.model_use_layernorm,
                selection_layers=args.multimodel_selection_layers,
                selection_hidden_features=args.multimodel_selection_hidden_features,
                lerp_value=args.multimodel_selection_lerp,
                selection_mode=args.multimodel_selection_mode,
                clustering_feature_size=clustering_feature_size,
                num_top_outputs=args.multimodel_num_top_outputs,
                first_stage_epochs=args.multimodel_first_stage_epochs,
                shared_first_layers=args.multimodel_shared_first_layers,
                shared_last_layers=args.multimodel_shared_last_layers,
            ).to(self.device)
        elif args.model == ADAPTIVE_RESNET:
            model = adaptive_resnet.AdaptiveResnet(
                in_features=in_features,
                out_features=out_features,
                output_every=args.model_output_every,
                layers=args.model_layers,
                hidden_features=args.model_width,
                use_layernorm=args.model_use_layernorm
            ).to(self.device)
        elif args.model == LEARNED_RAY:
            model = learned_ray.LearnedRayModel(
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                use_layernorm=args.model_use_layernorm
            ).to(self.device)
        elif args.model == ATLASNET:
            model = atlasnet.AtlasNet(
                in_features=in_features,
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                atlas_layers=args.atlasnet_atlas_layers,
                atlas_features=args.atlasnet_atlas_features
            ).to(self.device)
        elif args.model == SUBNET:
            subnet_factors = [
                (x[0], x[1]) for x in args.subnet_factors] if args.subnet_factors else None
            if args.subnet_interleave:
                raise ValueError("Interleave deprecated")
            model = subnet.SubNet(
                in_features=in_features,
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                factors=subnet_factors
            ).to(self.device)
        elif args.model == SUBNET_SVD:
            subnet_factors = [
                (x[0], x[1]) for x in args.subnet_factors] if args.subnet_factors else None
            if args.subnet_interleave:
                raise ValueError("Interleave deprecated")
            model = subnet_svd.SubNetSVD(
                in_features=in_features,
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                factors=subnet_factors,
                svd_components=args.svd_components,
                load_from=args.svd_load_from
            ).to(self.device)
        elif args.model == SVDNET:
            subnet_factors = args.subnet_factors if args.subnet_factors else None
            model = svdnet.SVDNet(
                in_features=in_features,
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                factors=subnet_factors,
                svd_components=args.svd_components,
                load_from=args.svd_load_from
            ).to(self.device)
        elif args.model == MULTISUBNET:
            model = multisubnet.MultiSubnet(
                num_models=args.multimodel_num_models,
                in_features=in_features,
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                selection_layers=args.multimodel_selection_layers,
                selection_hidden_features=args.multimodel_selection_hidden_features,
                lerp_value=args.multimodel_selection_lerp,
                selection_mode=args.multimodel_selection_mode
            ).to(self.device)
        elif args.model == MULTIRESNET:
            model = multiresnet.MultiResnet(
                num_models=args.multimodel_num_models,
                in_features=in_features,
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                selection_layers=args.multimodel_selection_layers,
                selection_hidden_features=args.multimodel_selection_hidden_features,
                lerp_value=args.multimodel_selection_lerp,
                selection_mode=args.multimodel_selection_mode,
                output_every=args.model_output_every
            ).to(self.device)
        elif args.model == MIPNET:
            subnet_factors = [
                (x[0], x[1]) for x in args.subnet_factors] if args.subnet_factors else None
            model = mipnet.MipNet(
                in_features=in_features,
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                factors=subnet_factors,
                share_gradients=args.mipnet_share_gradients
            ).to(self.device)
        elif args.model == SMIPNET:
            subnet_factors = [
                (x[0], x[1]) for x in args.subnet_factors] if args.subnet_factors else None
            model = smipnet.SMipNet(
                in_features=in_features,
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                factors=subnet_factors
            ).to(self.device)
        elif args.model == PRUNINGMLP:
            model = pruning_mlp.PruningMLP(in_features=in_features,
                                           out_features=out_features,
                                           layers=args.model_layers,
                                           hidden_features=args.model_width,
                                           use_layernorm=args.model_use_layernorm,
                                           load_from=args.svd_load_from).to(self.device)
        elif args.model == SLIMNET:
            model = slimnet.SlimNet(
                in_features=in_features,
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                min_factor=args.slimnet_min_factor,
                share_gradients=args.mipnet_share_gradients,
                fixed_layers=args.lcslimnet_fixed_layers,
                masking_continuous_lod=args.lcslimnet_lodmasking
            ).to(self.device)
        elif args.model == STREAMABLE:
            initial_width = 128
            final_width = args.model_width
            current_width = initial_width if args.script_mode == options.TRAIN else final_width
            model = streamable_nf.ProgressiveSiren(
                in_feats=in_features,
                hidden_feats=current_width,
                n_hidden_layers=args.model_layers - 1,
                out_feats=out_features,
                device=self.device,
                use_relu=True
            ).to(self.device)
        elif args.model == LCNET:
            model = lcnet.LCNet(
                in_features=in_features,
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                use_layernorm=args.model_use_layernorm,
                latent_size=64,
                num_lods=4
            ).to(self.device)
        elif args.model == MULTIMIPNET:
            subnet_factors = [
                (x[0], x[1]) for x in args.subnet_factors] if args.subnet_factors else None
            model = multimipnet.MultiMipnet(
                num_models=args.multimodel_num_models,
                in_features=in_features,
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                selection_layers=args.multimodel_selection_layers,
                selection_hidden_features=args.multimodel_selection_hidden_features,
                lerp_value=args.multimodel_selection_lerp,
                selection_mode=args.multimodel_selection_mode,
                factors=subnet_factors,
            ).to(self.device)
        elif args.model == LCMIPNET:
            subnet_factors = [
                (x[0], x[1]) for x in args.subnet_factors] if args.subnet_factors else None
            model = lcmipnet.LCMipNet(
                in_features=in_features,
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                factors=subnet_factors,
                share_gradients=args.mipnet_share_gradients,
                latent_size=64
            ).to(self.device)
        elif args.model == LCSLIMNET:
            subnet_factors = [
                (x[0], x[1]) for x in args.subnet_factors] if args.subnet_factors else None
            model = lcslimnet.LCSlimNet(
                in_features=in_features,
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                min_factor=args.slimnet_min_factor,
                default_factors=subnet_factors,
                share_gradients=args.mipnet_share_gradients,
                latent_size=args.lcslimnet_latent_size,
                latent_codes_lods=args.lcslimnet_latent_codes_lods,
                init_same_latent_codes=args.lcslimnet_init_same_latent_codes,
                fixed_layers=args.lcslimnet_fixed_layers,
                one_lc_per_lod=args.lcslimnet_one_lc_per_lod,
                masking_continuous_lod=args.lcslimnet_lodmasking
            ).to(self.device)
        elif args.model == MULTISLNET:
            clustering_feature_size = base_feature_size + out_features
            model = multislnet.MultiSLNet(
                num_models=args.multimodel_num_models,
                in_features=in_features,
                out_features=out_features,
                layers=args.model_layers,
                hidden_features=args.model_width,
                use_layernorm=args.model_use_layernorm,
                selection_layers=args.multimodel_selection_layers,
                selection_hidden_features=args.multimodel_selection_hidden_features,
                lerp_value=args.multimodel_selection_lerp,
                selection_mode=args.multimodel_selection_mode,
                clustering_feature_size=clustering_feature_size,
                first_stage_epochs=args.multimodel_first_stage_epochs,
                shared_first_layers=args.multimodel_shared_first_layers,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model {args.model}")
        aux_model = None
        if args.use_aux_network:
            aux_out_features = 2 if args.aux_encode_saliency else 1
            aux_model = mlp.MLP(in_features=in_features,
                                out_features=aux_out_features,
                                layers=args.aux_layers,
                                hidden_features=args.aux_features,
                                use_layernorm=args.aux_layernorm).to(self.device)
        importance_model = None
        if args.use_importance_training:
            importance_model = mlp.MLP(in_features=in_features,
                                       out_features=1,
                                       layers=args.importance_layers,
                                       hidden_features=args.importance_features).to(self.device)
        latent_code_layer = torch.nn.Identity()
        if args.use_latent_codes:
            num_latent_codes = len(self.train_data)
            if args.num_latent_codes >= 0:
                num_latent_codes = args.num_latent_codes
            else:
                num_latent_codes = num_latent_codes // np.abs(
                    args.num_latent_codes)
            latent_code_layer = video_layers.LatentCodeLayer(
                num_latent_codes,
                len(self.train_data) - 1,
                args.latent_code_dim).to(self.device)
        return model, latent_code_layer, importance_model, aux_model

    def setup_checkpoints(self):
        args = self.args

        checkpoint_manager = torch_checkpoint_manager.CheckpointManager(
            args.checkpoints_dir,
            max_to_keep=args.checkpoint_count)
        writer = None
        if args.script_mode == options.TRAIN:
            writer = SummaryWriter(log_dir=os.path.join(
                args.checkpoints_dir, "logs"))
        model_parameters = list(self.model.parameters())
        if args.use_importance_training:
            model_parameters.extend(self.importance_model.parameters())
        if args.use_aux_network:
            model_parameters.extend(self.aux_model.parameters())
        optimizer = torch.optim.Adam(model_parameters, lr=args.learning_rate)
        latent_optimizer = None
        if args.use_latent_codes:
            latent_optimizer = torch.optim.Adam(
                self.latent_code_layer.parameters(), lr=args.latent_learning_rate)
        latest_checkpoint = checkpoint_manager.load_latest_checkpoint()
        lr_scheduler = None
        if args.model != STREAMABLE:
            lr_scheduler = (torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_schedule_gamma)
                            if args.lr_schedule_gamma > 0.0 else None)
        if latest_checkpoint:
            self.model.load_state_dict(
                latest_checkpoint[CheckpointKeys.MODEL_STATE_DICT])
            self.latent_code_layer.load_state_dict(
                latest_checkpoint[CheckpointKeys.LATENT_CODE_STATE_DICT])
            optimizer.load_state_dict(
                latest_checkpoint[CheckpointKeys.OPTIMIZER_STATE_DICT])
            if args.use_latent_codes:
                latent_optimizer.load_state_dict(
                    latest_checkpoint[CheckpointKeys.LATENT_OPTIMIZER_STATE_DICT])
            if self.importance_model:
                self.importance_model.load_state_dict(
                    latest_checkpoint[CheckpointKeys.IMPORTANCE_MODEL_STATE_DICT])
            if self.aux_model:
                self.aux_model.load_state_dict(
                    latest_checkpoint[CheckpointKeys.AUX_MODEL_STATE_DICT])

        # Load the aux model from the other checkpoint if it is available.
        if not latest_checkpoint and args.svd_load_from:
            load_from_checkpoint = torch_checkpoint_manager.CheckpointManager(
                args.svd_load_from).load_latest_checkpoint()
            if load_from_checkpoint and CheckpointKeys.AUX_MODEL_STATE_DICT in load_from_checkpoint and self.aux_model:
                self.aux_model.load_state_dict(
                    load_from_checkpoint[CheckpointKeys.AUX_MODEL_STATE_DICT])

        return checkpoint_manager, writer, optimizer, latent_optimizer, lr_scheduler

    def get_importance_map(self, height: int, width: int, focal: float, pose: torch.Tensor, inference_height: int = 64,
                           t: float = 0.0) -> torch.Tensor:
        """Performs inference on the importance map network to yield an importance map.

        Args:
            height: Final height of the importance map.
            width: Final width of the importance map.
            focal: Focal of the importance map.
            pose: Target pose.
            inference_height: Internal height.
            t: Time.

        Returns:
            The importance map.

        """
        args = self.args
        if isinstance(self.importance_model, torch.nn.Module):
            inference_width = int(round(inference_height * (width / height)))
            my_focal = focal * (inference_height / height)
            ray_origins, ray_directions = nerf_utils.get_ray_bundle(
                inference_height, inference_width, my_focal, pose)
            if args.model == LEARNED_RAY:
                rays_plucker = self.model.ray_to_indices(
                    ray_origins, ray_directions)
            else:
                rays_plucker = my_torch_utils.convert_rays_to_plucker(
                    ray_origins, ray_directions)
            rays_with_t = torch.cat(
                (rays_plucker, t * torch.ones_like(rays_plucker[:, :, :1])), dim=-1).reshape(-1, 7)
            rays_positional_encoding = nerf_utils.add_positional_encoding(
                rays_with_t[:, :-1],
                num_encoding_functions=args.positional_encoding_functions)
            rays_with_positional = torch.cat(
                (rays_positional_encoding, rays_with_t[:, -1:]), dim=-1)
            if args.use_latent_codes:
                features = self.latent_code_layer(rays_with_positional)
            else:
                features = rays_with_positional
            importance_map_lowres = self.importance_model(
                features).reshape(inference_height, inference_width)
            importance_map = F.interpolate(importance_map_lowres[None, None], (height, width), mode="bilinear",
                                           align_corners=False)
            raw_importance_map = torch.sigmoid(
                importance_map[0, 0, :, :, None].reshape(-1))
            return my_utils.lerp(1.0, raw_importance_map, args.importance_lerp)
        else:
            raise ValueError("Unrecognized type of importance map.")

    def render_full_image(self, batch_size: int, rays: torch.Tensor,
                          reference_image: Optional[torch.Tensor] = None,
                          early_stopping: bool = False,
                          lods: Optional[Sequence[int]] = None,
                          aux_discarding: bool = False,
                          extra_options: Dict[Any, Any] = None) -> RenderFullImageOutputs:
        """Renders a full image.

        Args:
            batch_size: Batch size to use.
            rays: Full set of rays to use.
            reference_image: Reference image.
            early_stopping: Use early stopping.
            lods: level of detail.
            aux_discarding: Discard pixels using aux network.
            extra_options: Extra options.

        Returns:
            The final render output.
        """
        height, width, n_features = rays.shape
        iterations = int(np.ceil(height * width / batch_size))
        all_ray_outputs = []
        expected_layers_outputs = []
        layer_outputs = []
        mm_model_selection = []
        mm_model_top_selection_indices = []
        uvmap_outputs = []
        aux_outputs = []
        iterable = range(iterations) if iterations <= 1 else tqdm.tqdm(
            range(iterations), leave=False)
        for i in iterable:
            ray_subset = rays.reshape((height * width, n_features))[
                (i * batch_size):(
                    (i + 1) * batch_size)]
            reference_image_subset = None
            if reference_image is not None:
                reference_image_subset = reference_image.reshape(
                    (height * width, 3))[(i * batch_size):((i + 1) * batch_size)]
            run_outputs = self.do_inference(ray_subset,
                                            early_stopping=early_stopping,
                                            ray_colors=reference_image_subset,
                                            lods=lods,
                                            aux_discarding=aux_discarding,
                                            extra_options=extra_options)
            model_output = run_outputs.model_output
            if isinstance(model_output, adaptive_mlp.AdaptiveMLPOutputs):
                all_ray_outputs.append(
                    model_output.expected_output)
                expected_layers_outputs.append(
                    model_output.expected_layers)
                if model_output.layer_outputs is not None:
                    layer_outputs.append(
                        model_output.layer_outputs)
            elif isinstance(model_output, multimodel.MultiModelMLPOutputs):
                if aux_discarding:
                    output_tensor = model_output.model_output
                    selection_tensor = model_output.selection_indices
                    padded_model_output = torch.zeros(
                        (ray_subset.shape[0], output_tensor.shape[1]),
                        device=output_tensor.device,
                        dtype=output_tensor.dtype)
                    padded_model_output[run_outputs.aux_discarding_keep] = output_tensor
                    all_ray_outputs.append(padded_model_output)
                    full_selection_output = torch.zeros(
                        (ray_subset.shape[0]),
                        device=selection_tensor.device,
                        dtype=selection_tensor.dtype)
                    full_selection_output[run_outputs.aux_discarding_keep] = selection_tensor
                    mm_model_selection.append(full_selection_output)
                    if model_output.top_selection_indices is not None:
                        tsi_tensor = model_output.top_selection_indices
                        tsi_0 = torch.zeros(
                            (ray_subset.shape[0], tsi_tensor[0].shape[1]),
                            device=tsi_tensor[0].device,
                            dtype=tsi_tensor[0].dtype)
                        tsi_0[run_outputs.aux_discarding_keep] = tsi_tensor[0]
                        tsi_1 = torch.zeros(
                            (ray_subset.shape[0], tsi_tensor[0].shape[1]),
                            device=tsi_tensor[1].device,
                            dtype=tsi_tensor[1].dtype)
                        tsi_1[run_outputs.aux_discarding_keep] = tsi_tensor[1]
                        mm_model_top_selection_indices.append((tsi_0, tsi_1))
                else:
                    all_ray_outputs.append(model_output.model_output)
                    mm_model_selection.append(model_output.selection_indices)
                    if model_output.top_selection_indices is not None:
                        mm_model_top_selection_indices.append(
                            model_output.top_selection_indices)
            elif (isinstance(model_output, adaptive_resnet.AdaptiveResnetOutputs) or
                  isinstance(model_output, subnet.SubNetOutputs) or
                  isinstance(model_output, mipnet.MipNetOutputs) or
                  isinstance(model_output, subnet_svd.SubNetSVDOutputs) or
                  isinstance(model_output, svdnet.SVDNetOutputs) or
                  isinstance(model_output, smipnet.SMipNetOutputs) or
                  isinstance(model_output, slimnet.SlimNetOutputs) or
                  isinstance(model_output, lcnet.LCNetOutputs) or
                  isinstance(model_output, lcmipnet.LCMipNetOutputs) or
                  isinstance(model_output, lcslimnet.LCSlimNetOutputs)):
                if aux_discarding:
                    output_tensor = model_output.outputs
                    padded_model_output = torch.zeros(
                        (ray_subset.shape[0], output_tensor.shape[1],
                         output_tensor.shape[2]),
                        device=output_tensor.device,
                        dtype=output_tensor.dtype)
                    padded_model_output[run_outputs.aux_discarding_keep] = output_tensor
                    all_ray_outputs.append(padded_model_output[:, -1])
                    layer_outputs.append(padded_model_output)
                else:
                    all_ray_outputs.append(model_output.outputs[:, -1])
                    layer_outputs.append(model_output.outputs)
            elif isinstance(model_output, torch.Tensor):
                if aux_discarding:
                    padded_model_output = torch.zeros((ray_subset.shape[0], model_output.shape[1]),
                                                      device=model_output.device,
                                                      dtype=model_output.dtype)
                    padded_model_output[run_outputs.aux_discarding_keep] = model_output
                    all_ray_outputs.append(padded_model_output)
                else:
                    all_ray_outputs.append(model_output)
            elif isinstance(model_output, atlasnet.AtlasNetOutputs):
                all_ray_outputs.append(model_output.model_output)
                uvmap_outputs.append(model_output.uv_map)
            elif (isinstance(model_output, multisubnet.MultiSubnetOutputs) or
                  isinstance(model_output, multiresnet.MultiResnetOutputs) or
                  isinstance(model_output, multimipnet.MultiMipnetOutputs) or
                  isinstance(model_output, multislnet.MultiSLNetOutputs)):
                if aux_discarding:
                    output_tensor = model_output.model_outputs
                    selection_tensor = model_output.selection_indices
                    padded_model_output = torch.zeros(
                        (ray_subset.shape[0], output_tensor.shape[2]),
                        device=output_tensor.device,
                        dtype=output_tensor.dtype)
                    padded_model_output[run_outputs.aux_discarding_keep] = output_tensor[:, -1]
                    all_ray_outputs.append(padded_model_output)
                    full_layer_output = torch.zeros(
                        (ray_subset.shape[0], output_tensor.shape[1],
                         output_tensor.shape[2]),
                        device=output_tensor.device,
                        dtype=output_tensor.dtype)
                    full_layer_output[run_outputs.aux_discarding_keep] = output_tensor
                    layer_outputs.append(full_layer_output)
                    full_selection_output = torch.zeros(
                        (ray_subset.shape[0]),
                        device=selection_tensor.device,
                        dtype=selection_tensor.dtype)
                    full_selection_output[run_outputs.aux_discarding_keep] = selection_tensor
                    mm_model_selection.append(full_selection_output)
                else:
                    all_ray_outputs.append(model_output.model_outputs[:, -1])
                    layer_outputs.append(model_output.model_outputs)
                    mm_model_selection.append(model_output.selection_indices)
            else:
                raise ValueError(f"Unknown type {type(model_output)}")
            if run_outputs.aux_output is not None:
                aux_outputs.append(run_outputs.aux_output)
        all_ray_outputs = all_ray_outputs[0] if len(
            all_ray_outputs) == 1 else torch.cat(all_ray_outputs, dim=0)
        all_ray_outputs = all_ray_outputs.reshape(
            (height, width, 3 if not self.args.predict_alpha else 4))
        outputs = RenderFullImageOutputs(all_ray_outputs)
        if expected_layers_outputs:
            outputs.expected_layers_image = torch.cat(
                expected_layers_outputs, dim=0).reshape(height, width, 1)
        if layer_outputs:
            outputs.layer_outputs = layer_outputs[0] if len(
                layer_outputs) == 1 else torch.cat(layer_outputs, dim=0)
            outputs.layer_outputs = outputs.layer_outputs.permute((1, 0, 2)).reshape(-1, height, width,
                                                                                     self.model.out_features)
        if mm_model_selection:
            outputs.mm_model_selection = torch.cat(
                mm_model_selection, dim=0).reshape(height, width)
        if mm_model_top_selection_indices:
            k = mm_model_top_selection_indices[0][0].shape[1]
            outputs.mm_model_top_selection_indices = (
                torch.cat([x[0] for x in mm_model_top_selection_indices],
                          dim=0).reshape(height, width, k),
                torch.cat([x[1] for x in mm_model_top_selection_indices],
                          dim=0).reshape(height, width, k)
            )
        if uvmap_outputs:
            outputs.uvmap_outputs = torch.cat(uvmap_outputs, dim=0).reshape(
                height, width, uvmap_outputs[0].shape[-1])
        if aux_outputs:
            outputs.aux_image = torch.cat(
                aux_outputs, dim=0).reshape(height, width, -1)
        if self.args.render_truncate_alpha > 0.0:
            truncate_val = self.args.render_truncate_alpha
            truncate_tensor = torch.tensor(
                (False, False, False, True), dtype=torch.bool, device=self.device)
            tf_mask = (0.0 * outputs.full_image).bool()
            tf_mask[outputs.full_image[:, :, 3]
                    < truncate_val] = truncate_tensor
            outputs.full_image[tf_mask] = truncate_val * \
                torch.pow(outputs.full_image[tf_mask] / truncate_val, 3.0)
            if layer_outputs:
                tf_mask = (0.0 * outputs.layer_outputs).bool()
                tf_mask[outputs.layer_outputs[:, :, :, 3]
                        < truncate_val] = truncate_tensor
                outputs.layer_outputs[tf_mask] = (
                    truncate_val * torch.pow(outputs.layer_outputs[tf_mask] / truncate_val, 3.0))
        return outputs

    def render_full_image_multilod(self, batch_size: int, rays: torch.Tensor,
                                   lodmap: torch.Tensor = None,
                                   reference_image: Optional[torch.Tensor] = None,
                                   early_stopping: bool = False,
                                   aux_discarding: bool = False,
                                   extra_options: Dict[Any, Any] = None) -> RenderFullImageOutputs:
        """Renders a full image with per-pixel LoD.

        Args:
            batch_size: Batch size to use.
            lodmap: Level of detail per pixel.
            rays: Full set of rays to use.
            reference_image: Reference image.
            early_stopping: Use early stopping.
            aux_discarding: Discard pixels using aux network.
            extra_options: Extra options.

        Returns:
            The final render output.
        """
        height, width, n_features = rays.shape
        iterations = int(np.ceil(height * width / batch_size))
        all_ray_outputs = []
        expected_layers_outputs = []
        layer_outputs = []
        mm_model_selection = []
        uvmap_outputs = []
        aux_outputs = []
        iterable = range(iterations) if iterations <= 1 else tqdm.tqdm(
            range(iterations), leave=False)
        for i in iterable:
            ray_subset = rays.reshape((height * width, n_features))[
                (i * batch_size):(
                    (i + 1) * batch_size)]
            lodmap_subset = lodmap.reshape(
                (height * width))[(i * batch_size):((i + 1) * batch_size)]
            reference_image_subset = None
            if reference_image is not None:
                reference_image_subset = reference_image.reshape(
                    (height * width, 3))[(i * batch_size):((i + 1) * batch_size)]
            unique_lods = torch.unique(lodmap_subset).tolist()
            lod_to_run_output = {}
            model_output = None
            run_outputs = None
            for lod in unique_lods:
                run_outputs = self.do_inference(ray_subset[lodmap_subset == lod],
                                                early_stopping=early_stopping,
                                                ray_colors=reference_image_subset,
                                                lods=[lod],
                                                aux_discarding=aux_discarding,
                                                extra_options=extra_options)
                model_output = run_outputs.model_output
                lod_to_run_output[lod] = run_outputs
            if (isinstance(model_output, adaptive_resnet.AdaptiveResnetOutputs) or
                    isinstance(model_output, subnet.SubNetOutputs) or
                    isinstance(model_output, mipnet.MipNetOutputs) or
                    isinstance(model_output, subnet_svd.SubNetSVDOutputs) or
                    isinstance(model_output, svdnet.SVDNetOutputs) or
                    isinstance(model_output, smipnet.SMipNetOutputs) or
                    isinstance(model_output, slimnet.SlimNetOutputs) or
                    isinstance(model_output, lcslimnet.LCSlimNetOutputs)):
                if aux_discarding:
                    padded_model_output = torch.zeros(
                        (ray_subset.shape[0], model_output.outputs.shape[1],
                         model_output.outputs.shape[2]),
                        device=model_output.outputs.device,
                        dtype=model_output.outputs.dtype)
                    for lod in unique_lods:
                        m_run_output = lod_to_run_output[lod]
                        m_sub = torch.eq(lodmap_subset, lod)
                        m_sub[m_sub.clone()] = m_run_output.aux_discarding_keep
                        padded_model_output[m_sub] = m_run_output.model_output.outputs
                        # Proper way of doing this with a view:
                        # padded_model_output[lodmap_subset == lod][
                        #     m_run_output.aux_discarding_keep] = m_run_output.model_output.outputs
                    all_ray_outputs.append(padded_model_output[:, -1])
                    layer_outputs.append(padded_model_output)
                else:
                    padded_model_output = torch.zeros(
                        (ray_subset.shape[0], model_output.outputs.shape[1],
                         model_output.outputs.shape[2]),
                        device=model_output.outputs.device,
                        dtype=model_output.outputs.dtype)
                    for lod in unique_lods:
                        m_run_output = lod_to_run_output[lod]
                        padded_model_output[lodmap_subset ==
                                            lod] = m_run_output.model_output.outputs
                    all_ray_outputs.append(padded_model_output[:, -1])
                    layer_outputs.append(padded_model_output)
            else:
                raise ValueError(f"Unknown type {type(model_output)}")
            if run_outputs.aux_output is not None:
                full_aux_output = torch.zeros(
                    (ray_subset.shape[0], run_outputs.aux_output.shape[1]),
                    device=model_output.outputs.device,
                    dtype=model_output.outputs.dtype)
                for lod in unique_lods:
                    m_run_output = lod_to_run_output[lod]
                    full_aux_output[lodmap_subset ==
                                    lod] = m_run_output.aux_output
                aux_outputs.append(full_aux_output)
        all_ray_outputs = torch.cat(all_ray_outputs, dim=0).reshape(
            (height, width, 3 if not self.args.predict_alpha else 4))
        outputs = RenderFullImageOutputs(all_ray_outputs)
        if expected_layers_outputs:
            outputs.expected_layers_image = torch.cat(
                expected_layers_outputs, dim=0).reshape(height, width, 1)
        if layer_outputs:
            outputs.layer_outputs = torch.cat(layer_outputs, dim=0).permute((1, 0, 2)).reshape(
                -1,
                height, width, self.model.out_features)
        if mm_model_selection:
            outputs.mm_model_selection = torch.cat(
                mm_model_selection, dim=0).reshape(height, width)
        if uvmap_outputs:
            outputs.uvmap_outputs = torch.cat(uvmap_outputs, dim=0).reshape(
                height, width, uvmap_outputs[0].shape[-1])
        if aux_outputs:
            outputs.aux_image = torch.cat(
                aux_outputs, dim=0).reshape(height, width, -1)
        if self.args.render_truncate_alpha > 0.0:
            truncate_val = self.args.render_truncate_alpha
            truncate_tensor = torch.tensor(
                (False, False, False, True), dtype=torch.bool, device=self.device)
            tf_mask = (0.0 * outputs.full_image).bool()
            tf_mask[outputs.full_image[:, :, 3]
                    < truncate_val] = truncate_tensor
            outputs.full_image[tf_mask] = truncate_val * \
                torch.pow(outputs.full_image[tf_mask] / truncate_val, 3.0)
            if layer_outputs:
                tf_mask = (0.0 * outputs.layer_outputs).bool()
                tf_mask[outputs.layer_outputs[:, :, :, 3]
                        < truncate_val] = truncate_tensor
                outputs.layer_outputs[tf_mask] = (
                    truncate_val * torch.pow(outputs.layer_outputs[tf_mask] / truncate_val, 3.0))
        return outputs

    def do_validation_run(self, step: int):
        """Do a single validation run.

        Args:
            step: Current step.

        Returns:
            None.
        """
        args = self.args
        writer = self.writer
        device = self.device

        with torch.no_grad():
            if args.validation_interval >= 0 and (step == 1 or
                                                  args.validation_interval == 0 or
                                                  step % args.validation_interval == 0):
                self.model.eval()
                batch_size = args.val_batch_size
                render_poses = self.train_data.get_render_poses(
                    args.dataset_render_poses)

                val_images = []
                val_expected_layers_images = []
                val_mm_selection_image = []
                val_images_es = []
                val_images_layers_es = []
                val_layers = []
                render_image_times = []
                render_image_times_es = []
                aux_images = []
                for i in tqdm.tqdm(range(args.dataset_render_poses), desc="Val", leave=False):
                    t = 0.0
                    height = self.train_data.height
                    width = self.train_data.width
                    if self.train_data.using_processed_camera_parameters:
                        ray_origins, ray_directions = self.train_data.camera_params_to_rays(
                            render_poses["intrinsics"][i].to(self.device),
                            render_poses["transforms"][i].to(self.device),
                            height, width
                        )
                    else:
                        focal = self.train_data.focal
                        pose = render_poses[i]
                        ray_origins, ray_directions = nerf_utils.get_ray_bundle(height, width, focal,
                                                                                pose[:3, :4].to(device))
                    if args.model == LEARNED_RAY:
                        rays_plucker = self.model.ray_to_indices(
                            ray_origins, ray_directions)
                    else:
                        rays_plucker = my_torch_utils.convert_rays_to_plucker(
                            ray_origins, ray_directions)
                    rays_with_t = torch.cat(
                        (rays_plucker, t *
                         torch.ones_like(rays_plucker[:, :, :1])),
                        dim=-1)
                    t0 = time.time()
                    rendered_image_outputs = self.render_full_image(
                        batch_size, rays_with_t)
                    render_image_times.append(time.time() - t0)
                    val_images.append(rendered_image_outputs.full_image.cpu())
                    if i == 0 and rendered_image_outputs.layer_outputs is not None:
                        val_layers.append(
                            rendered_image_outputs.layer_outputs.cpu())
                    if rendered_image_outputs.aux_image is not None:
                        aux_images.append(
                            rendered_image_outputs.aux_image.cpu())
                    if args.model == ADAPTIVE_MLP:
                        expected_layers_image = expected_layers_image / \
                            (len(self.model.model_layers) - 1)
                        val_expected_layers_images.append(
                            expected_layers_image)
                        t0 = time.time()
                        rendered_image_es_outputs = self.render_full_image(batch_size,
                                                                           rays_with_t,
                                                                           early_stopping=True)
                        render_image_times_es.append(time.time() - t0)
                        val_images_es.append(
                            rendered_image_es_outputs.full_image)
                        expected_layers_image_es = rendered_image_es_outputs.expected_layers_image / (
                            len(self.model.model_layers) - 1)
                        val_images_layers_es.append(expected_layers_image_es)
                    if rendered_image_outputs.mm_model_selection is not None:
                        val_mm_selection_image.append(
                            rendered_image_outputs.mm_model_selection.cpu())
                val_images = torch.stack(val_images)
                writer.add_scalar("val/10_render_time",
                                  np.mean(render_image_times), step)
                writer.add_images("val/10_output_images", torch.clamp(val_images[:, :, :, :3], 0, 1), step,
                                  dataformats="NHWC")
                if args.predict_alpha:
                    writer.add_images(
                        "val/11_output_alpha",
                        val_images[:, :, :, 3:4].clamp(0, 1),
                        step, dataformats="NHWC")
                writer.add_video("val/30_output_video",
                                 val_images[None, :, :, :, :3].permute(
                                     (0, 1, 4, 2, 3)).clamp(0, 1),
                                 step,
                                 fps=args.video_framerate)
                if args.model == ADAPTIVE_MLP:
                    writer.add_scalar(
                        "val/10_render_time_earlystopping", np.mean(render_image_times_es), step)
                    val_expected_layers_images = torch.stack(
                        val_expected_layers_images)
                    writer.add_images("val/11_output_expected_layers_images", val_expected_layers_images, step,
                                      dataformats="NHWC")
                    val_images_es = torch.stack(val_images_es)
                    writer.add_images(
                        "val/20_output_images_es", val_images_es[:, :, :, :3], step, dataformats="NHWC")
                    val_images_layers_es = torch.stack(val_images_layers_es)
                    writer.add_images("val/21_output_layers_images_es",
                                      val_images_layers_es, step, dataformats="NHWC")
                if args.model in ADAPTIVE_NETWORKS:
                    if val_layers:
                        writer.add_images("val/31_output_layers_images", torch.clamp(val_layers[0][:, :, :, :3], 0, 1),
                                          step, dataformats="NHWC")
                if val_mm_selection_image:
                    val_mm_selection_image = torch.stack(
                        val_mm_selection_image)[:, :, :, None]
                    val_mm_selection_image = my_torch_utils.greyscale_to_turbo_colormap(
                        val_mm_selection_image / args.multimodel_num_models)
                    writer.add_images(
                        "val/30_multimodel_selection", val_mm_selection_image, step, dataformats="NHWC")
                if aux_images:
                    aux_images = torch.stack(aux_images)
                    writer.add_images(
                        "val/40_aux_images", aux_images[:,
                                                        :, :, :1].clamp(0, 1),
                        step, dataformats="NHWC")
                    if args.aux_encode_saliency:
                        writer.add_images(
                            "val/40_aux_saliency", aux_images[:, :, :, 1:2], step, dataformats="NHWC")
                self.model.train()

    def compute_validation_psnr(self):
        """Computes the validation PSNR.
        """
        args = self.args
        device = self.device
        cropped_psnr_values = [[]
                               for _ in range(self.model.num_outputs)]
        if self.val_data is None:
            raise ValueError("No validation data")
        with torch.no_grad():
            self.model.eval()
            data_loader = DataLoader(self.val_data, batch_size=1, shuffle=False,
                                     num_workers=args.dataloader_num_workers)
            for frame_num, data in enumerate(tqdm.tqdm(data_loader, desc="Validation_PSNR", leave=False)):
                frame_batch_size, height, width, data_image_channels = data["image"].shape
                ray_colors = data["image"].to(device)
                ray_masks = data["mask"].to(
                    device) if "mask" in data else None
                ray_t = (torch.zeros_like(
                    data["image"][:, :, :, 0]) + data["t"][:, None, None]).to(device)
                ray_origins = []
                ray_directions = []
                for i in range(frame_batch_size):
                    if self.train_data.using_processed_camera_parameters:
                        transform = data["transform"][i].to(device)
                        intrinsics = data["intrinsics"][i].to(device)
                        frame_ray_origins, frame_ray_directions = self.train_data.camera_params_to_rays(
                            intrinsics, transform, height, width)
                    else:
                        pose_target = data["pose"][0, :3, :4]
                        focal = self.train_data.focal
                        frame_ray_origins, frame_ray_directions = nerf_utils.get_ray_bundle(height, width,
                                                                                            focal,
                                                                                            pose_target)
                    ray_origins.append(frame_ray_origins)
                    ray_directions.append(frame_ray_directions)
                ray_origins = torch.cat(ray_origins)
                ray_directions = torch.cat(ray_directions)
                batch_size = args.val_batch_size
                if args.model == LEARNED_RAY:
                    rays_plucker = self.model.ray_to_indices(
                        ray_origins, ray_directions)
                else:
                    rays_plucker = my_torch_utils.convert_rays_to_plucker(
                        ray_origins, ray_directions)
                rays_with_t = torch.cat(
                    (rays_plucker, ray_t[0, :, :, None]), dim=-1)
                for lod in tqdm.tqdm(range(self.model.num_outputs), desc="LoDs", leave=False):
                    rendered_image_outputs = self.render_full_image(
                        batch_size, rays_with_t, early_stopping=True, lods=[lod])
                    cropped_psnr_val = pytorch_psnr.cropped_psnr(
                        ray_colors[:1, :, :, :].permute((0, 3, 1, 2)),
                        rendered_image_outputs.full_image[None, :, :, :].permute(
                            (0, 3, 1, 2))
                    )
                    cropped_psnr_values[lod].append(
                        float(cropped_psnr_val))
        self.model.train()
        avg_cropped_psnr_values = [
            np.mean(x) for x in cropped_psnr_values]
        return avg_cropped_psnr_values[0]

    def save_checkpoint(self, step: int, force: bool = False):
        """Saves a checkpoint.

        Args:
            step: The current step.
            force: Force a checkpoint
        """
        args = self.args
        if args.checkpoint_interval >= 0 and (args.checkpoint_interval == 0 or
                                              step % args.checkpoint_interval == 0 or force):
            data_dict = {
                CheckpointKeys.MODEL_STATE_DICT: self.model.state_dict(),
                CheckpointKeys.LATENT_CODE_STATE_DICT: self.latent_code_layer.state_dict(),
                CheckpointKeys.OPTIMIZER_STATE_DICT: self.optimizer.state_dict(),
            }
            if args.use_latent_codes:
                data_dict[CheckpointKeys.LATENT_OPTIMIZER_STATE_DICT] = self.latent_optimizer.state_dict()
            if args.use_importance_training:
                data_dict[CheckpointKeys.IMPORTANCE_MODEL_STATE_DICT] = self.importance_model.state_dict()
            if args.use_aux_network:
                data_dict[CheckpointKeys.AUX_MODEL_STATE_DICT] = self.aux_model.state_dict()
            self.checkpoint_manager.save_checkpoint(data_dict)
            self.writer.flush()

    def log_training_to_tensorboard(self, step: int, run_outputs: InferenceOutputs,
                                    other_variables: dict,
                                    epoch: Optional[int] = None):
        """Logs training to tensorboard.

        Args:
            step: The current global step.
            run_outputs: Outputs of the run.
            other_variables: Other variables to log.
            epoch: The current epoch.

        Returns:
            None
        """
        args = self.args
        writer = self.writer
        run_outputs_items = [(f.name, getattr(run_outputs, f.name))
                             for f in dataclasses.fields(run_outputs)]
        for k, v in itertools.chain(run_outputs_items, other_variables.items()):
            if isinstance(v, float) or isinstance(v, int):
                writer.add_scalar("train/" + k, v, step)
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                writer.add_scalar("train/" + k, v.item(), step)
        if epoch is not None:
            writer.add_scalar("train/epoch", epoch, step)

        if (args.train_tensorboard_interval >= 0 and step % args.train_tensorboard_interval == 0):
            with torch.no_grad():
                if "gt_image" in other_variables:
                    self.writer.add_image("train/gt_image", other_variables["gt_image"][0],
                                          self.checkpoint_manager.step,
                                          dataformats="HWC")

    def do_inference(self, rays: torch.Tensor,
                     early_stopping: bool = False,
                     ray_colors: Optional[torch.Tensor] = None,
                     ray_mask: Optional[torch.Tensor] = None,
                     ray_saliency: Optional[torch.Tensor] = None,
                     ray_mipcolors: Optional[torch.Tensor] = None,
                     lods: Optional[Sequence[int]] = None,
                     aux_discarding: bool = False,
                     extra_options: Dict[Any, Any] = None) -> InferenceOutputs:
        """Do a forward pass for inference and compute losses.

        Args:
            rays: Set of rays as an (N, D+1) tensor with t in last index.
            early_stopping: Use early stopping.
            ray_colors: Ground truth colors as (N, 3) tensor.
            ray_mask: Optional value for mask at each ray.
            ray_saliency: Ray saliency
            ray_mipcolors: Mipped colors.
            lods: Lods to render.
            aux_discarding: Discard pixels using aux network.
            extra_options: Dictionary of options to pass.

        Returns:
            A dict containing the run outputs.
        """
        extra_options = {} if extra_options is None else extra_options
        args = self.args
        if args.sh_encoding_degree > 0:
            raise NotImplementedError()
        elif args.positional_encoding_functions:
            rays_positional_encoding = nerf_utils.add_positional_encoding(
                rays[:, :-1],
                num_encoding_functions=args.positional_encoding_functions)
            rays_with_encoding = torch.cat(
                (rays_positional_encoding, rays[:, -1:]), dim=-1)
        else:
            rays_with_encoding = rays
        if args.use_latent_codes:
            features = self.latent_code_layer(rays_with_encoding)
        else:
            features = rays_with_encoding
        aux_model_output = None
        aux_discarding_keep = None
        if args.use_aux_network:
            aux_model_output: torch.Tensor = self.aux_model(features)
            if aux_discarding:
                aux_discarding_keep = (
                    aux_model_output[:, 0] > args.aux_discard_threshold)
                features = features[aux_discarding_keep]
        if args.model == ADAPTIVE_MLP:
            model_output: adaptive_mlp.AdaptiveMLPOutputs = self.model(
                features, early_stopping)
        elif args.model == MULTIMODEL:
            model_output: multimodel.MultiModelMLPOutputs = self.model(
                features)
        elif args.model == ADAPTIVE_RESNET:
            if lods:
                model_output: adaptive_resnet.AdaptiveResnetOutputs = self.model(
                    features, lods)
            else:
                model_output: adaptive_resnet.AdaptiveResnetOutputs = self.model(
                    features)
        elif args.model in (MLP, RESMLP, LEARNED_RAY, PRUNINGMLP):
            model_output: torch.Tensor = self.model(features)
        elif args.model == ATLASNET:
            model_output: atlasnet.AtlasNetOutputs = self.model(features)
        elif args.model == SUBNET:
            if lods:
                model_output: subnet.SubNetOutputs = self.model(features, [self.model.factors[x] for x in lods],
                                                                **extra_options)
            elif self.model.training and args.subnet_optimized_training:
                model_output: subnet.SubNetOutputs = self.model(features, [
                    self.rng.choice(self.model.factors[:-1]),
                    self.model.factors[-1],
                ], **extra_options)
            else:
                model_output: subnet.SubNetOutputs = self.model(
                    features, **extra_options)
        elif args.model == MIPNET:
            if lods:
                factors = [self.model.factors[x] +
                           (self.model.factors[x - 1] if x > 0 else (0, 0,)) for x in lods]
                model_output: mipnet.MipNetOutputs = self.model(
                    features, factors)
            else:
                model_output: mipnet.MipNetOutputs = self.model(features)
        elif args.model == SMIPNET:
            if lods:
                factors = [self.model.factors[x] for x in lods]
                model_output: smipnet.SMipNetOutputs = self.model(
                    features, factors)
            else:
                model_output: smipnet.SMipNetOutputs = self.model(features)
        elif args.model == MULTISUBNET:
            model_output: multisubnet.MultiSubnetOutputs = self.model(features)
        elif args.model == MULTIRESNET:
            model_output: multiresnet.MultiResnetOutputs = self.model(features)
        elif args.model == SUBNET_SVD:
            if lods:
                model_output: subnet_svd.SubNetSVDOutputs = self.model(features, [self.model.factors[x] for x in lods],
                                                                       **extra_options)
            elif self.model.training and args.subnet_optimized_training:
                model_output: subnet_svd.SubNetSVDOutputs = self.model(features, [
                    self.rng.choice(self.model.factors[:-1]),
                    self.model.factors[-1],
                ], **extra_options)
            else:
                model_output: subnet_svd.SubNetSVDOutputs = self.model(
                    features, **extra_options)
        elif args.model == SVDNET:
            if lods:
                model_output: svdnet.SVDNetOutputs = self.model(features,
                                                                [self.model.factors[x]
                                                                    for x in lods],
                                                                **extra_options)
            elif self.model.training and args.subnet_optimized_training:
                model_output: svdnet.SVDNetOutputs = self.model(features, [
                    self.rng.choice(self.model.factors[:-1]),
                    self.model.factors[-1],
                ], **extra_options)
            else:
                model_output: svdnet.SVDNetOutputs = self.model(
                    features, **extra_options)
        elif args.model == LCNET:
            if lods:
                model_output = self.model(features, lods)
            else:
                model_output = self.model(features)
        elif args.model == STREAMABLE:
            model_output: torch.Tensor = self.model(features)
        elif args.model == MULTIMIPNET:
            if lods:
                factors = [self.model.factors[x] +
                           (self.model.factors[x - 1] if x > 0 else (0, 0,)) for x in lods]
                model_output: multimipnet.MultiMipnetOutputs = self.model(
                    features, factors)
            else:
                model_output: multimipnet.MultiMipnetOutputs = self.model(
                    features)
        elif args.model in (SLIMNET, LCMIPNET, LCSLIMNET, MULTISLNET):
            model_output = self.model(features, lods)
        else:
            raise ValueError("Unknown model")
        loss = None
        color_loss = None
        efficiency_loss = None
        loadbalance_loss = None
        clustering_loss = None
        mm_importance_loss = None
        lc_loss = None
        aux_loss = None
        if ray_colors is not None:
            if args.use_importance_training:
                raise NotImplementedError()
            gt = ray_colors if args.predict_alpha else ray_colors[:, :3]
            if args.predict_alpha and gt.shape[1] < 4:
                gt = torch.cat((ray_colors, ray_mask[:, None]), dim=-1)
            if args.model == ADAPTIVE_MLP and not early_stopping:
                layer_outputs = model_output.layer_outputs
                layer_stop_here_probabilities = model_output.layer_stop_here_probabilities
                if args.lossfn_color == "l1":
                    pixel_loss = torch.abs(layer_outputs - gt[:, None, :])
                elif args.lossfn_color == "l2":
                    pixel_loss = torch.pow(layer_outputs - gt[:, None, :], 2.0)
                else:
                    raise ValueError("Unknown loss function")
                color_loss = torch.mean(
                    pixel_loss * layer_stop_here_probabilities[:, :, None])
                efficiency_loss = torch.mean(model_output.expected_layers)
                loss = (color_loss +
                        args.efficiency_loss_lambda * efficiency_loss)
            elif args.model == MULTIMODEL:
                if args.lossfn_color == "l1":
                    pixel_loss = torch.abs(model_output.model_output - gt)
                    color_loss = torch.mean(pixel_loss)
                elif args.lossfn_color == "l2":
                    pixel_loss = torch.pow(model_output.model_output - gt, 2.0)
                    color_loss = torch.mean(pixel_loss)
                else:
                    raise ValueError("Unknown loss function")
                per_ray_loss = torch.mean(pixel_loss, dim=-1)
                if args.multimodel_importance_loss_lambda > 1e-10:
                    mm_selection_probability = torch.log(torch.gather(model_output.selection_probabilities, 1,
                                                                      model_output.selection_indices[:, None]))[:, 0]
                    mm_importance_loss = args.multimodel_importance_loss_lambda * torch.mean(
                        -mm_selection_probability * -per_ray_loss.detach())
                if args.multimodel_loadbalance_loss_lambda > 1e-10:
                    loadbalance_loss = args.multimodel_loadbalance_loss_lambda * multimodel.compute_load_balance_loss(model_output.selection_probabilities,
                                                                                                                      model_output.selection_indices)
                if args.multimodel_clustering_loss_lambda > 1e-10:
                    expected_features = model_output.selection_probabilities @ self.model.clustering_features
                    num_ray_features = 6
                    channels = gt.shape[1]
                    if args.multimodel_clustering_loss_version == 0:
                        position_diff = torch.linalg.norm(
                            expected_features[:, :num_ray_features] - rays[:, :num_ray_features])
                        color_diff = torch.linalg.norm(
                            expected_features[:, num_ray_features:(num_ray_features+channels)] - gt)
                        clustering_loss = args.multimodel_clustering_loss_lambda * (
                            position_diff + color_diff)
                    elif args.multimodel_clustering_loss_version == 1:
                        ray_dir_cos = torch.mean(F.cosine_similarity(
                            expected_features[:, :3], rays[:, :3]))
                        ray_momentum_cos = torch.mean(F.cosine_similarity(
                            expected_features[:, 3:6], rays[:, 3:6]))
                        ray_momentum_diff_norm = torch.mean(torch.linalg.norm(
                            expected_features[:, 3:6] - rays[:, 3:6], dim=1))
                        color_diff = torch.mean(torch.linalg.norm(
                            expected_features[:, num_ray_features:(num_ray_features+channels)] - gt, dim=1))
                        clustering_loss = args.multimodel_clustering_loss_lambda * (
                            -ray_dir_cos - ray_momentum_cos + ray_momentum_diff_norm + color_diff)
                    elif args.multimodel_clustering_loss_version == 2:
                        color_diff = torch.mean(torch.linalg.norm(
                            expected_features[:, num_ray_features:(num_ray_features+channels)] - gt, dim=1))
                        clustering_loss = args.multimodel_clustering_loss_lambda * color_diff
                    else:
                        raise NotImplementedError(
                            f"Unknown clustering loss verison {args.multimodel_clustering_loss_version}")
                loss = (color_loss +
                        (mm_importance_loss or 0.0) +
                        (loadbalance_loss or 0.0) +
                        (clustering_loss or 0.0))
            elif args.model in (ADAPTIVE_RESNET, SUBNET, SUBNET_SVD, SVDNET):
                if args.lossfn_color == "l1":
                    pixel_loss = torch.abs(model_output.outputs - gt[:, None])
                elif args.lossfn_color == "l2":
                    pixel_loss = torch.pow(
                        model_output.outputs - gt[:, None], 2.0)
                else:
                    raise ValueError("Unknown loss function")
                if ray_mask is not None and args.lossfn_color_mask_factor > 0:
                    color_loss = torch.mean(
                        pixel_loss * (1.0 + args.lossfn_color_mask_factor * ray_mask[:, None]))
                else:
                    color_loss = torch.mean(pixel_loss)
                loss = (color_loss)
            elif args.model in (MIPNET, SMIPNET, SLIMNET, LCNET, LCMIPNET, LCSLIMNET):
                m_gt = gt[:, None] if ray_mipcolors is None else ray_mipcolors
                if args.lossfn_color == "l1":
                    pixel_loss = torch.abs(model_output.outputs - m_gt)
                elif args.lossfn_color == "l2":
                    pixel_loss = torch.pow(model_output.outputs - m_gt, 2.0)
                else:
                    raise ValueError("Unknown loss function")
                if ray_mask is not None and args.lossfn_color_mask_factor > 0:
                    color_loss = torch.mean(
                        pixel_loss * (1.0 + args.lossfn_color_mask_factor * ray_mask[:, None]))
                else:
                    color_loss = torch.mean(pixel_loss)
                if args.model in (LCNET, LCMIPNET, LCSLIMNET):
                    lc_norm = torch.linalg.vector_norm(
                        self.model.latent_codes, ord=1, dim=1)
                    lc_loss = args.lossfn_lc_factor * \
                        torch.mean(torch.pow(lc_norm - 1, 2.0))
                loss = (color_loss + (0 if lc_loss is None else lc_loss))
            elif args.model in (MLP, RESMLP, LEARNED_RAY, PRUNINGMLP, STREAMABLE):
                if args.lossfn_color == "l1":
                    pixel_loss = torch.abs(model_output - gt)
                elif args.lossfn_color == "l2":
                    pixel_loss = torch.pow(model_output - gt, 2.0)
                else:
                    raise ValueError("Unknown loss function")
                if ray_mask is not None and args.lossfn_color_mask_factor > 0:
                    color_loss = torch.mean(
                        pixel_loss * (1.0 + args.lossfn_color_mask_factor * ray_mask[:, None]))
                else:
                    color_loss = torch.mean(pixel_loss)
                loss = (color_loss)
            elif args.model == ATLASNET:
                if args.lossfn_color == "l1":
                    pixel_loss = torch.abs(model_output.model_output - gt)
                elif args.lossfn_color == "l2":
                    pixel_loss = torch.pow(model_output.model_output - gt, 2.0)
                else:
                    raise ValueError("Unknown loss function")
                if ray_mask is not None and args.lossfn_color_mask_factor > 0:
                    color_loss = torch.mean(
                        pixel_loss * (1.0 + args.lossfn_color_mask_factor * ray_mask[:, None]))
                else:
                    color_loss = torch.mean(pixel_loss)
                loss = (color_loss)
            elif args.model in (MULTISUBNET, MULTIRESNET, MULTIMIPNET, MULTISLNET):
                m_gt = ray_mipcolors if (ray_mipcolors is not None and args.model in (
                    MULTIMIPNET, MULTISLNET)) else gt[:, None]
                if args.lossfn_color == "l1":
                    pixel_loss = torch.abs(
                        model_output.model_outputs - m_gt)
                    color_loss = torch.mean(pixel_loss)
                elif args.lossfn_color == "l2":
                    pixel_loss = torch.pow(
                        model_output.model_outputs - m_gt, 2.0)
                    color_loss = torch.mean(pixel_loss)
                else:
                    raise ValueError("Unknown loss function")
                per_ray_loss = torch.mean(pixel_loss[:, -1], dim=-1)
                mm_selection_probability = torch.log(torch.gather(model_output.selection_probabilities, 1,
                                                                  model_output.selection_indices[:, None]))[:, 0]
                mm_importance_loss = torch.mean(
                    -mm_selection_probability * -per_ray_loss.detach())
                loadbalance_loss = multimodel.compute_load_balance_loss(model_output.selection_probabilities,
                                                                        model_output.selection_indices)
                loss = (color_loss +
                        args.multimodel_importance_loss_lambda * mm_importance_loss +
                        args.multimodel_loadbalance_loss_lambda * loadbalance_loss)
            else:
                raise ValueError(f"Unknown model {str(type(self.model))}")
            if args.use_aux_network:
                aux_truth = torch.gt(ray_mask, 1 / 255).float()[:, None]
                if args.aux_encode_saliency:
                    aux_truth = torch.cat(
                        (aux_truth, ray_saliency[:, None]), dim=1)
                aux_loss = torch.pow(aux_model_output - aux_truth, 2.0)
                aux_loss = torch.mean(aux_loss)
                loss += aux_loss
        return InferenceOutputs(
            model_output=model_output,
            loss=loss,
            color_loss=color_loss,
            efficiency_loss=efficiency_loss,
            loadbalance_loss=loadbalance_loss,
            clustering_loss=clustering_loss,
            lc_loss=lc_loss,
            mm_importance_loss=mm_importance_loss,
            aux_loss=aux_loss,
            aux_output=aux_model_output,
            ray_colors=ray_colors,
            aux_discarding_keep=aux_discarding_keep
        )

    def run_training_loop(self):
        """Runs the primary training loop."""
        args = self.args
        device = self.device
        train_start_time = time.time()

        if args.use_latent_codes and self.checkpoint_manager.step == 0:
            self.writer.add_scalar("train/num_latent_codes",
                                   self.latent_code_layer.num_latent_codes,
                                   self.checkpoint_manager.step)

        self.model.train()
        data_loader = DataLoader(self.train_data, batch_size=args.frame_batch_size, shuffle=True,
                                 num_workers=args.dataloader_num_workers,
                                 prefetch_factor=args.dataloader_prefetch_factor)
        current_lod = None
        max_lod = (-1 if args.slimnet_progressive_training else (
            self.model.num_outputs - 1))
        for epoch in tqdm.tqdm(range(args.epochs), desc="Epoch"):
            if hasattr(self.model, "set_epoch"):
                self.model.set_epoch(epoch)
            if args.slimnet_progressive_training:
                max_lod = min(max_lod+1, self.model.num_outputs - 1)
            for i, (e, lod, factor) in enumerate(args.lod_factor_schedule):
                if epoch == e:
                    current_lod = lod
                    if not args.lod_factor_schedule_use_mip:
                        self.train_data.update_factor(factor)
                        data_loader = DataLoader(self.train_data, batch_size=args.frame_batch_size, shuffle=True,
                                                 num_workers=args.dataloader_num_workers,
                                                 prefetch_factor=args.dataloader_prefetch_factor)
                    if args.model == STREAMABLE:
                        new_width = int(
                            round(args.model_width * args.subnet_factors[lod][0]))
                        self.model.runnable_widths.append(new_width)
                        if epoch != 0:
                            old_width = 128 if lod == 1 else int(
                                round(args.model_width * args.subnet_factors[lod-1][0]))
                            tqdm.tqdm.write(f"Growing width to {new_width}")
                            self.model.grow_width(new_width - old_width)
                        self.model.select_subnet(i)
                        model_parameters = list(self.model.parameters())
                        if args.use_importance_training:
                            model_parameters.extend(
                                self.importance_model.parameters())
                        if args.use_aux_network:
                            model_parameters.extend(
                                self.aux_model.parameters())
                        self.optimizer = torch.optim.Adam(
                            model_parameters, lr=args.learning_rate)
                    break
            for data in tqdm.tqdm(data_loader, desc="Frame Batch", leave=False):
                frame_batch_size, height, width, data_image_channels = data["image"].shape
                ray_colors = data["image"].reshape(-1, data_image_channels)
                ray_masks = data["mask"].reshape(
                    -1) if "mask" in data else None
                ray_saliency = data["saliency"].reshape(
                    -1) if "saliency" in data else None
                ray_t = (torch.zeros_like(
                    data["image"][:, :, :, 0]) + data["t"][:, None, None]).reshape(-1)
                ray_uvs = torch.meshgrid(
                    torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing='ij')
                ray_uvs = torch.stack(
                    (ray_uvs[1], ray_uvs[0]), dim=-1)[None].expand(frame_batch_size, -1, -1, -1)
                ray_uvs_imindex = torch.linspace(-1, 1,
                                                    frame_batch_size)[:, None, None, None]
                ray_uvs_imindex = ray_uvs_imindex.expand(
                    -1, height, width, 1)
                ray_uvs = torch.cat(
                    (ray_uvs, ray_uvs_imindex), dim=3).reshape(-1, 3)
                ray_origins = []
                ray_directions = []
                data_mip_images_cuda = {
                    f"mip_image_{x}": data[f"mip_image_{x}"].permute((3, 0, 1, 2))[None].to(device)
                    for x in range(self.model.num_outputs - 1) if f"mip_image_{x}" in data
                }
                if (args.model in (SLIMNET, LCSLIMNET) or 
                    (args.model in (MIPNET, LCNET, MULTIMIPNET, LCMIPNET, MULTISLNET) and args.mipnet_use_sat)):
                    image_sat = sat_utils.build_sat_image(
                        data["image"].to(device), args.sat_sample_floatingpoint)
                for i in range(frame_batch_size):
                    if self.train_data.using_processed_camera_parameters:
                        transform = data["transform"][i].to(device)
                        intrinsics = data["intrinsics"][i].to(device)
                        frame_ray_origins, frame_ray_directions = self.train_data.camera_params_to_rays(
                            intrinsics, transform, height, width)
                    else:
                        pose_target = data["pose"][0, :3, :4]
                        focal = self.train_data.focal
                        frame_ray_origins, frame_ray_directions = nerf_utils.get_ray_bundle(height, width, focal,
                                                                                            pose_target)
                    ray_origins.append(
                        frame_ray_origins.reshape(-1, 3).cpu())
                    ray_directions.append(
                        frame_ray_directions.reshape(-1, 3).cpu())
                ray_origins = torch.cat(ray_origins)
                ray_directions = torch.cat(ray_directions)
                if args.saliency_sampling_factor >= 0:
                    ray_probabilities = my_utils.lerp(
                        1, ray_saliency.to(device), args.saliency_sampling_factor)
                    if args.saliency_sampling_gradient_factor > 0:
                        image_gradients = my_torch_utils.compute_image_gradients(
                            data["image"].to(device))
                        image_gradients = image_gradients.abs().mean(dim=3).reshape(-1)
                        ray_probabilities = ray_probabilities + \
                            args.saliency_sampling_gradient_factor * image_gradients
                        del image_gradients
                    if args.saliency_sampling_mask_factor > 0:
                        ray_probabilities = ray_probabilities + \
                            args.saliency_sampling_mask_factor * \
                            ray_masks.to(device)
                    if False:
                        # Export ray probabilities image.
                        ray_probabilities_image = ray_probabilities.reshape(
                            2, height,  width)
                        my_torch_utils.save_torch_image(
                            "temp/ray_probabilities.png", ray_probabilities_image[0])
                    ray_probabilities_cdf = torch.cumsum(
                        ray_probabilities, dim=0)
                    ray_probabilities_cdf_max = ray_probabilities_cdf[-1]
                    foreground_pixels = torch.sum(
                        ray_masks.to(device) >= 0.1)
                    num_pixels_to_sample = int(
                        foreground_pixels / (1 - args.subset_bg_pixels))
                    ray_ordering = torch.searchsorted(
                        ray_probabilities_cdf,
                        ray_probabilities_cdf_max * torch.rand(
                            num_pixels_to_sample, device=ray_probabilities_cdf.device))
                    ray_ordering = ray_ordering.cpu()
                    iterations = int(np.ceil(len(ray_ordering) /
                                                args.batch_size) * args.train_frame_factor)
                    if False:
                        # Export ray ordering image.
                        sampled_rays = torch.zeros(
                            (height, width), dtype=torch.float32)
                        sampled_rays.ravel()[
                            ray_ordering[ray_ordering < height * width]] = 1
                        my_torch_utils.save_torch_image(
                            "temp/sampled_rays.png", sampled_rays)
                elif args.saliency_sampling_factor_v2 >= 0:
                    background_pixels = ray_masks.to(device) < 0.1
                    foreground_pixels = torch.logical_not(
                        background_pixels).nonzero(as_tuple=True)[0]
                    background_pixels = background_pixels.nonzero(as_tuple=True)[
                        0]
                    # Sample background pixels.
                    fg_saliency = ray_saliency.to(
                        device)[foreground_pixels]
                    fg_probabilities = my_utils.lerp(
                        1, fg_saliency, args.saliency_sampling_factor_v2)
                    fg_probabilities_cdf = torch.cumsum(
                        fg_probabilities, dim=0)
                    fg_probabilities_cdf_max = fg_probabilities_cdf[-1]
                    sampled_fg_pixels_indices = torch.searchsorted(
                        fg_probabilities_cdf,
                        fg_probabilities_cdf_max * torch.rand(
                            len(foreground_pixels), device=fg_probabilities_cdf.device))
                    sampled_fg_pixels = foreground_pixels[sampled_fg_pixels_indices]
                    bg_pixels_per_batch = int(
                        args.subset_bg_pixels * args.batch_size)
                    fg_pixels_per_batch = int(
                        args.batch_size - bg_pixels_per_batch)
                    num_batches = min(len(sampled_fg_pixels) // fg_pixels_per_batch,
                                        len(background_pixels) // bg_pixels_per_batch)
                    foreground_pixels_indices = torch.randperm(
                        len(sampled_fg_pixels), device=device)
                    foreground_pixels_ar = sampled_fg_pixels[
                        foreground_pixels_indices[:fg_pixels_per_batch * num_batches]]
                    foreground_pixels_ar = foreground_pixels_ar.reshape(
                        num_batches, fg_pixels_per_batch)
                    background_pixels_indices = torch.randperm(
                        len(background_pixels), device=device)
                    background_pixels_ar = background_pixels[
                        background_pixels_indices[:bg_pixels_per_batch * num_batches]]
                    background_pixels_ar = background_pixels_ar.reshape(
                        num_batches, bg_pixels_per_batch)
                    ray_ordering = torch.cat(
                        (foreground_pixels_ar, background_pixels_ar), dim=1).reshape(-1)
                    ray_ordering = torch.cat(
                        (ray_ordering,
                            sampled_fg_pixels[foreground_pixels_indices[
                                fg_pixels_per_batch * num_batches:fg_pixels_per_batch * (num_batches + 1)]],
                            background_pixels[background_pixels_indices[
                                bg_pixels_per_batch * num_batches:bg_pixels_per_batch * (num_batches + 1)]]))
                    iterations = int(np.ceil(len(ray_ordering) /
                                                args.batch_size) * args.train_frame_factor)
                elif args.subset_bg_pixels >= 0 and ray_masks is not None:
                    background_pixels = ray_masks.to(device) < 0.1
                    foreground_pixels = torch.logical_not(
                        background_pixels).nonzero(as_tuple=True)[0]
                    background_pixels = background_pixels.nonzero(as_tuple=True)[
                        0]
                    bg_pixels_per_batch = int(
                        args.subset_bg_pixels * args.batch_size)
                    fg_pixels_per_batch = int(
                        args.batch_size - bg_pixels_per_batch)
                    num_batches = min(len(foreground_pixels) // fg_pixels_per_batch,
                                        len(background_pixels) // bg_pixels_per_batch)
                    foreground_pixels_indices = torch.randperm(
                        len(foreground_pixels), device=device)
                    foreground_pixels_ar = foreground_pixels[
                        foreground_pixels_indices[:fg_pixels_per_batch * num_batches]]
                    foreground_pixels_ar = foreground_pixels_ar.reshape(
                        num_batches, fg_pixels_per_batch)
                    background_pixels_indices = torch.randperm(
                        len(background_pixels), device=device)
                    background_pixels_ar = background_pixels[
                        background_pixels_indices[:bg_pixels_per_batch * num_batches]]
                    background_pixels_ar = background_pixels_ar.reshape(
                        num_batches, bg_pixels_per_batch)
                    ray_ordering = torch.cat(
                        (foreground_pixels_ar, background_pixels_ar), dim=1).reshape(-1)
                    ray_ordering = torch.cat(
                        (ray_ordering,
                            foreground_pixels[foreground_pixels_indices[
                                fg_pixels_per_batch * num_batches:fg_pixels_per_batch * (num_batches + 1)]],
                            background_pixels[background_pixels_indices[
                                bg_pixels_per_batch * num_batches:bg_pixels_per_batch * (num_batches + 1)]]))
                    iterations = int(np.ceil(len(ray_ordering) /
                                                args.batch_size) * args.train_frame_factor)
                else:
                    if epoch == 0:
                        tqdm.tqdm.write(
                            "Warning: Not using fg/bg sampling")
                    ray_ordering = self.rng.choice(frame_batch_size * height * width,
                                                    size=frame_batch_size * height * width,
                                                    replace=False)
                    iterations = int(np.ceil(len(ray_ordering) /
                                                args.batch_size) * args.train_frame_factor)
                for i in tqdm.tqdm(range(iterations), desc="Ray Batch",
                                   leave=False):
                    step = self.checkpoint_manager.increment_step()
                    self.optimizer.zero_grad()
                    self.latent_optimizer.zero_grad() if self.latent_optimizer else None
                    importance_map = None
                    if args.use_importance_training:
                        raise NotImplementedError()
                    else:
                        indices = ray_ordering[
                            (i * args.batch_size):((i + 1) * args.batch_size)].cpu()
                    ray_origins_batch = ray_origins[indices].to(device)
                    ray_directions_batch = ray_directions[indices].to(device)
                    ray_colors_batch = ray_colors[indices].to(device)
                    ray_masks_batch = ray_masks[indices].to(
                        device) if ray_masks is not None else None
                    ray_t_batch = ray_t[indices].to(device)
                    ray_saliency_batch = ray_saliency[indices].to(
                        device) if ray_saliency is not None else None
                    ray_mipcolors_batch = None
                    lods = None if current_lod is None else [current_lod]
                    if args.continuous_lod_training:
                        lods = [self.rng.uniform(0, self.model.num_outputs - 1),
                                self.model.num_outputs - 1]
                        if hasattr(self.model, 'lod_to_scale'):
                            scale = self.model.lod_to_scale(
                                lods[0]) * args.sat_scale_multiplier
                        else:
                            # Hardcoded so 0=8, 1=4, 2=2, 3=1
                            scale = 2**(3-lods[0]) * args.sat_scale_multiplier
                        if args.sat_sample_floatingpoint:
                            lod_ray_batch = sat_utils.sample_from_sat_fp(
                                image_sat, ray_uvs[indices].to(device), scale
                            )
                        else:
                            lod_ray_batch = sat_utils.sample_from_sat(
                                image_sat, ray_uvs[indices].to(device), scale)
                        ray_mipcolors_batch = torch.stack(
                            (lod_ray_batch, ray_colors_batch), dim=1)
                    elif args.model in (MIPNET, SMIPNET, LCNET, MULTIMIPNET, LCMIPNET, MULTISLNET) and args.mipnet_use_sat:
                        lods = [self.rng.choice(
                            self.model.num_outputs - 1), self.model.num_outputs - 1]
                        if args.mipnet_use_sat_factors:
                            scale = args.mipnet_use_sat_factors[lods[0]]
                        else:
                            scale = self.model.lod_to_scale(lods[0])
                        scale *= args.sat_scale_multiplier
                        if args.sat_sample_floatingpoint:
                            lod_ray_batch = sat_utils.sample_from_sat_fp(
                                image_sat, ray_uvs[indices].to(device), scale
                            )
                        else:
                            lod_ray_batch = sat_utils.sample_from_sat(
                                image_sat, ray_uvs[indices].to(device), scale)
                        ray_mipcolors_batch = torch.stack(
                            (lod_ray_batch, ray_colors_batch), dim=1)
                    elif args.model in (MIPNET, SMIPNET, LCNET, MULTIMIPNET, LCMIPNET, MULTISLNET) and current_lod is None:
                        lods = [self.rng.choice(
                            self.model.num_outputs - 1), self.model.num_outputs - 1]
                        lod_ray_batch = F.grid_sample(
                            data_mip_images_cuda[f"mip_image_{lods[0]}"],
                            ray_uvs[indices][None, :,
                                             None, None, :].to(device),
                            mode="bilinear", align_corners=True)[0, :, :, 0, 0].permute((1, 0))
                        ray_mipcolors_batch = torch.stack(
                            (lod_ray_batch, ray_colors_batch), dim=1)
                    elif (args.model in (MIPNET, SMIPNET) and
                          current_lod is not None and
                          args.lod_factor_schedule_use_mip):
                        lods = [current_lod]
                        lod_ray_batch = F.grid_sample(
                            data_mip_images_cuda[f"mip_image_{lods[0]}"],
                            ray_uvs[indices][None, :,
                                             None, None, :].to(device),
                            mode="bilinear", align_corners=True)[0, :, :, 0, 0].permute((1, 0))
                        ray_mipcolors_batch = lod_ray_batch
                    elif args.model in (SLIMNET, LCSLIMNET):
                        lods = [self.rng.choice(
                            self.model.num_outputs - 1), max_lod]
                        scale = self.model.lod_to_scale(
                            lods[0]) * args.sat_scale_multiplier
                        if args.sat_sample_floatingpoint:
                            lod_ray_batch = sat_utils.sample_from_sat_fp(
                                image_sat, ray_uvs[indices].to(device), scale)
                        else:
                            lod_ray_batch = sat_utils.sample_from_sat(
                                image_sat, ray_uvs[indices].to(device), scale)
                        lod_ray_batch2 = ray_colors_batch
                        if args.slimnet_progressive_training:
                            scale2 = self.model.lod_to_scale(
                                lods[1]) * args.sat_scale_multiplier
                            if args.sat_sample_floatingpoint:
                                lod_ray_batch2 = sat_utils.sample_from_sat_fp(
                                    image_sat, ray_uvs[indices].to(device), scale2)
                            else:
                                lod_ray_batch2 = sat_utils.sample_from_sat(
                                    image_sat, ray_uvs[indices].to(device), scale2)
                        ray_mipcolors_batch = torch.stack(
                            (lod_ray_batch, lod_ray_batch2), dim=1)
                    if args.model == LEARNED_RAY:
                        rays_plucker = self.model.ray_to_indices(
                            ray_origins_batch, ray_directions_batch)
                    else:
                        rays_plucker = my_torch_utils.convert_rays_to_plucker(
                            ray_origins_batch, ray_directions_batch)
                    rays_subset_with_t = torch.cat(
                        (rays_plucker, ray_t_batch[:, None]), dim=-1)
                    run_outputs = self.do_inference(
                        rays_subset_with_t,
                        early_stopping=False,
                        ray_colors=ray_colors_batch,
                        ray_mask=ray_masks_batch,
                        ray_saliency=ray_saliency_batch,
                        ray_mipcolors=None if ray_mipcolors_batch is None else ray_mipcolors_batch,
                        lods=lods)
                    other_logging_variables = {
                        "width": width,
                        "height": height,
                        "importance_map": importance_map,
                    }
                    self.log_training_to_tensorboard(step, run_outputs, other_variables=other_logging_variables,
                                                     epoch=epoch)
                    loss = run_outputs.loss
                    loss.backward()
                    if args.model == STREAMABLE and current_lod > 0:
                        self.model.freeze_subnet(current_lod-1)
                    self.optimizer.step()
                    self.latent_optimizer.step() if self.latent_optimizer else None
                    self.do_validation_run(step)
                    self.save_checkpoint(step)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if args.model == MULTIMODEL and args.multimodel_selection_freeze_epochs == epoch:
                self.model.freeze_selection_model()
            if args.training_val_psnr_cutoff > 0 and epoch > 0 and epoch % 10 == 0:
                validation_psnr = self.compute_validation_psnr()
                self.writer.add_scalar("train/validation_psnr",
                                       validation_psnr,
                                       self.checkpoint_manager.step)
                if validation_psnr > args.training_val_psnr_cutoff:
                    break
        self.save_checkpoint(self.checkpoint_manager.step, force=True)
        train_end_time = time.time()
        train_elapsed_time = train_end_time - train_start_time
        train_time_output_file = os.path.join(
            args.checkpoints_dir, "train_time.json")
        with open(train_time_output_file, "w") as f:
            json.dump({"train_time": train_elapsed_time}, f, indent=2)

    def run_inference(self):
        """Generates images along the path from the loaded model."""
        args = self.args
        device = self.device
        output_dir = args.checkpoints_dir
        output_frames_dir = os.path.join(output_dir, "frames")
        my_utils.rmdir_sync(output_frames_dir)
        os.makedirs(output_frames_dir, exist_ok=True)
        ffmpeg_bin = shutil.which("ffmpeg")
        if args.inference_halves:
            self.model.half()
            self.aux_model.half()
            print("using half precision")
        burned_in = False
        lods = args.inference_lods if args.inference_lods else None

        with torch.no_grad():
            self.model.eval()
            batch_size = args.val_batch_size
            render_poses = self.train_data.get_render_poses(
                args.dataset_render_poses)
            render_times = []
            selected_render_poses = args.inference_indices or range(
                args.dataset_render_poses)
            for i in tqdm.tqdm(selected_render_poses, desc="Inference"):
                t = (self.train_data.num_frames - 1) * \
                    i / (args.dataset_render_poses - 1)
                height = self.train_data.height
                width = self.train_data.width
                if self.train_data.using_processed_camera_parameters:
                    intrinsics = render_poses["intrinsics"][i].to(self.device)
                    transforms = render_poses["transforms"][i].to(self.device)
                    if args.inference_halves:
                        intrinsics = intrinsics.half()
                        transforms = transforms.half()
                    ray_origins, ray_directions = self.train_data.camera_params_to_rays(
                        intrinsics,
                        transforms,
                        height, width
                    )
                else:
                    focal = self.train_data.focal
                    pose = render_poses[i]
                    ray_origins, ray_directions = nerf_utils.get_ray_bundle(height, width, focal,
                                                                            pose[:3, :4].to(device))
                if args.model == LEARNED_RAY:
                    rays_plucker = self.model.ray_to_indices(
                        ray_origins, ray_directions)
                else:
                    rays_plucker = my_torch_utils.convert_rays_to_plucker(
                        ray_origins, ray_directions)
                rays_with_t = torch.cat(
                    (rays_plucker, t *
                     torch.ones_like(rays_plucker[:, :, :1])),
                    dim=-1)
                if not burned_in:
                    self.render_full_image(batch_size,
                                           rays_with_t,
                                           lods=lods,
                                           early_stopping=True,
                                           aux_discarding=True)
                    burned_in = True
                torch.cuda.synchronize()
                t0 = time.time()
                rendered_image_outputs = self.render_full_image(batch_size,
                                                                rays_with_t,
                                                                lods=lods,
                                                                early_stopping=True,
                                                                aux_discarding=True)
                torch.cuda.synchronize()
                t1 = time.time()
                render_times.append(t1 - t0)
                if args.model in ADAPTIVE_NETWORKS:
                    for j in range(len(rendered_image_outputs.layer_outputs)):
                        level = lods[j] if lods else j
                        my_utils.join_and_make(
                            output_frames_dir, f"level{level}")
                        my_torch_utils.save_torch_image(
                            os.path.join(output_frames_dir,
                                         f"level{level}", "%05d.png" % i),
                            rendered_image_outputs.layer_outputs[j]
                        )
                else:
                    my_torch_utils.save_torch_image(
                        os.path.join(output_frames_dir, "%05d.png" % i),
                        rendered_image_outputs.full_image
                    )
                if rendered_image_outputs.mm_model_selection is not None:
                    my_utils.join_and_make(
                        output_frames_dir, "mm_model_selection")
                    mm_selection_image = rendered_image_outputs.mm_model_selection
                    mm_selection_image = my_torch_utils.greyscale_to_turbo_colormap(
                        mm_selection_image[None, :, :, None] / self.model.num_models)
                    my_torch_utils.save_torch_image(
                        os.path.join(output_frames_dir,
                                     "mm_model_selection", "%05d.png" % i),
                        mm_selection_image[0]
                    )
            if not args.skip_gif_generation and shutil.which(ffmpeg_bin):
                if args.model in ADAPTIVE_NETWORKS:
                    for j in range(rendered_image_outputs.layer_outputs.shape[0]):
                        level = lods[j] if lods else j
                        join_frames_args = [
                            ffmpeg_bin,
                            "-framerate", str(args.video_framerate),
                            "-i", os.path.join(output_frames_dir,
                                               f"level{level}", "%05d.png"),
                            "-filter_complex",
                            "[0]format=pix_fmts=yuva420p,split=2[bg][fg];[bg]drawbox=c=black@1:replace=1:t=fill[bg];[bg][fg]overlay=format=auto",
                            "-y", os.path.join(output_dir,
                                               f"level{level}_video.gif")
                        ]
                        subprocess.run(join_frames_args)
                else:
                    join_frames_args = [
                        ffmpeg_bin,
                        "-framerate", str(args.video_framerate),
                        "-i", os.path.join(output_frames_dir, "%05d.png"),
                        "-filter_complex",
                        "[0]format=pix_fmts=yuva420p,split=2[bg][fg];[bg]drawbox=c=black@1:replace=1:t=fill[bg];[bg][fg]overlay=format=auto",
                        "-y", os.path.join(output_dir, "video.gif")
                    ]
                    subprocess.run(join_frames_args)

            print(f"Average render time: {np.mean(render_times)}")

    def run_debug(self):
        """Debugging method."""
        args = self.args

    def run_render_lod_epi(self):
        """Debugging method."""
        import seaborn as sns
        import matplotlib.pyplot as plt
        args = self.args
        device = self.device
        output_dir = args.checkpoints_dir
        output_frames_dir = os.path.join(output_dir, "epi_image")
        # my_utils.rmdir_sync(output_frames_dir)
        os.makedirs(output_frames_dir, exist_ok=True)
        height = self.train_data.height
        width = self.train_data.width
        max_lod = self.model.num_outputs - 1

        epi_height = 385
        epi_line_y = 443
        epi_illustration_height = 5

        if "sida_simba_take1v3" in args.dataset_path:
            poses = [0]
            crops = np.array([(1621, 1977, 2813, 3030)])
        else:
            raise NotImplementedError("Unknown run dataset")

        # random_map = torch.rand((height, width), device=self.device)
        pose_num = poses[0]
        render_height = crops[0, 3] - crops[0, 1]
        render_width = crops[0, 2] - crops[0, 0]
        lodmap = torch.zeros((render_height, render_width),
                             device=self.device, dtype=torch.long) + max_lod

        if args.model not in ADAPTIVE_NETWORKS:
            raise ValueError("Model is not adaptive")

        with torch.no_grad():
            self.model.eval()
            batch_size = args.val_batch_size
            render_poses = self.train_data.get_render_poses(
                args.dataset_render_poses)

            t = 0
            if self.train_data.using_processed_camera_parameters:
                ray_origins, ray_directions = self.train_data.camera_params_to_rays(
                    render_poses["intrinsics"][pose_num].to(self.device),
                    render_poses["transforms"][pose_num].to(self.device),
                    height, width
                )
            else:
                focal = self.train_data.focal
                pose = render_poses[pose_num]
                ray_origins, ray_directions = nerf_utils.get_ray_bundle(height, width, focal,
                                                                        pose[:3, :4].to(device))
            if args.model == LEARNED_RAY:
                rays_plucker = self.model.ray_to_indices(
                    ray_origins, ray_directions)
            else:
                rays_plucker = my_torch_utils.convert_rays_to_plucker(
                    ray_origins, ray_directions)
            rays_with_t = torch.cat(
                (rays_plucker, t * torch.ones_like(rays_plucker[:, :, :1])),
                dim=-1)
            rays_with_t = rays_with_t[crops[0, 1]:crops[0, 3], crops[0, 0]:crops[0, 2], :]
            print("rays_with_t.shape", rays_with_t.shape)
            print("lodmap.shape", lodmap.shape)
            full_image_outputs = self.render_full_image_multilod(batch_size,
                                                                 rays_with_t, lodmap,
                                                                 aux_discarding=True)
            full_image = full_image_outputs.full_image
            full_image[epi_line_y-epi_illustration_height//2:epi_line_y + epi_illustration_height//2,
                       :, :] = torch.tensor([0.176,0.580,0.973, 1], dtype=torch.float32, device=self.device).view(1, 1, 4)
            my_torch_utils.save_torch_image(
                os.path.join(output_frames_dir, f"render.png"),
                full_image
            )

            rays_with_t_lod_epi = rays_with_t[epi_line_y:epi_line_y +
                                              1, :, :].expand(epi_height, -1, -1)
            # lodmap = torch.zeros((render_height, render_width), device=self.device, dtype=torch.long) + max_lod
            lodmap_lod_epi = torch.linspace(0, max_lod, epi_height, device=self.device, dtype=torch.float32)
            lodmap_lod_epi = torch.round(lodmap_lod_epi).long().view(-1, 1).expand(-1, render_width)
            print("rays_with_t_lod_epi.shape", rays_with_t_lod_epi.shape)
            print("lodmap_lod_epi.shape", lodmap_lod_epi.shape)

            lod_epi_outputs = self.render_full_image_multilod(batch_size,
                                                              rays_with_t_lod_epi, lodmap_lod_epi,
                                                              aux_discarding=True)
            lod_epi_image = lod_epi_outputs.full_image
            my_torch_utils.save_torch_image(
                os.path.join(output_frames_dir, f"lod_epi.png"),
                lod_epi_image)

    def run_render_slimnet_levels(self):
        """Method to render all levels of details from slimnet."""
        args = self.args
        device = self.device

        self.model.eval()
        batch_size = args.val_batch_size

        if args.use_aux_network:
            self.aux_model.half()
        self.model.half()

        with torch.no_grad():
            render_times = []
            render_poses = self.train_data.get_render_poses(
                args.dataset_render_poses)
            lods = None
            if args.model == SLIMNET:
                if args.lcslimnet_lodmasking:
                    lods = tqdm.tqdm(np.linspace(0, self.model.num_outputs-1, 7 * 50 + 1), desc="LOD")
                else:
                    lods = tqdm.trange(0, self.model.num_outputs, desc="LOD")
            elif args.model in (LCNET, LCMIPNET):
                lods = tqdm.tqdm(np.linspace(0, 3, 300), desc="LOD")
            elif args.model == LCSLIMNET:
                lods = tqdm.trange(0, self.model.num_outputs, desc="LOD")
                # lods = tqdm.tqdm(np.linspace(200, 225, 100), desc="LOD")
                # lods = tqdm.tqdm(np.linspace(225, 300, 300), desc="LOD")
                pass
            for lod_index, lod in enumerate(lods):
                if "StudioV2CaptureTest0805_max_light" in args.dataset_path:
                    poses = [10]
                    crops = np.array([(1645, 1446, 2310, 2899)])
                elif "DavidDataset0113Config4v3" in args.dataset_path:
                    poses = [55]
                    crops = np.array([(1637, 1232, 2365, 2821)])
                elif "MariaCamera0124Take2" in args.dataset_path:
                    poses = [31]
                    crops = np.array([(1843, 1500, 2373, 2963)])
                elif "DavidDataset0113Config5v4" in args.dataset_path:
                    poses = [14]
                    crops = np.array([(1659, 1450, 2278, 2945)])
                elif "sida_simba_take1v3" in args.dataset_path:
                    poses = [0]
                    crops = np.array([(1671, 1988, 2762, 3008)])
                elif "Sharon0505_05s" in args.dataset_path:
                    poses = [39]
                    crops = np.array([(1565, 1510, 2306, 3008)])
                elif "Sharon0505" in args.dataset_path:
                    poses = [69]
                    crops = np.array([(1540, 1387, 2348, 2907)])
                elif "BaltimoreFoodPrep0218Take2" in args.dataset_path:
                    poses = [28]
                    crops = np.array([(1635, 1282, 2481, 2983)])
                else:
                    print(f"Unknown dataset: {args.dataset_path}")
                    exit(1)
                if args.dataset_resize_factor != 1:
                    crops = crops // args.dataset_resize_factor
                iterable = tqdm.tqdm(poses, desc="Pose", leave=False) if len(
                    poses) > 1 else poses
                for i, pose_num in enumerate(iterable):
                    t = ((self.train_data.num_frames - 1) *
                         pose_num / (args.dataset_render_poses - 1))
                    height = self.train_data.height
                    width = self.train_data.width
                    if self.train_data.using_processed_camera_parameters:
                        intrinsics = render_poses["intrinsics"][pose_num].to(
                            self.device)
                        transforms = render_poses["transforms"][pose_num].to(
                            self.device)
                        if args.inference_halves:
                            intrinsics = intrinsics.half()
                            transforms = transforms.half()
                        ray_origins, ray_directions = self.train_data.camera_params_to_rays(
                            intrinsics,
                            transforms,
                            height, width
                        )
                    else:
                        focal = self.train_data.focal
                        pose = render_poses[pose_num]
                        ray_origins, ray_directions = nerf_utils.get_ray_bundle(height, width, focal,
                                                                                pose[:3, :4].to(device))
                    if args.model == LEARNED_RAY:
                        rays_plucker = self.model.ray_to_indices(
                            ray_origins, ray_directions)
                    else:
                        rays_plucker = my_torch_utils.convert_rays_to_plucker(
                            ray_origins, ray_directions)
                    rays_with_t = torch.cat(
                        (rays_plucker, t *
                         torch.ones_like(rays_plucker[:, :, :1])),
                        dim=-1)
                    my_crop = crops[i]
                    rays_with_t = (
                        rays_with_t[my_crop[1]:my_crop[3], my_crop[0]:my_crop[2]]).half()
                    t0 = time.time()
                    rendered_image_outputs = self.render_full_image(batch_size,
                                                                    rays_with_t,
                                                                    lods=[lod],
                                                                    early_stopping=True,
                                                                    aux_discarding=True)
                    render_time = time.time() - t0
                    render_times.append(render_time)
                    image = rendered_image_outputs.full_image.float().cpu().numpy()
                    output_frames_dir = os.path.join(
                        args.checkpoints_dir, "frames")
                    my_utils.join_and_make(
                        output_frames_dir, f"view{pose_num}")
                    output_filepath = os.path.join(output_frames_dir,
                                                   f"view{pose_num}", "%05d.png" % lod_index)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
                    num_rows_to_concatenate = int(
                        80 / args.dataset_resize_factor)
                    image = np.concatenate(
                        (image, np.zeros_like(image[:num_rows_to_concatenate])), axis=0)
                    height, width, _ = image.shape
                    font_scale = 2 * (1 / args.dataset_resize_factor)
                    color = (0, 0, 0, 1)
                    font_thickness = int(
                        max(1, 5 * (1 / args.dataset_resize_factor)))
                    text = f"LOD {lod+1:.2f}"
                    text_size, _ = cv2.getTextSize(
                        text, font, font_scale, font_thickness)
                    text_w, text_h = text_size
                    org = (int(0.5 * width - 0.5 * text_w),
                           int(height-0.5 * text_h))
                    image = cv2.putText(image, text, org, font,
                                        font_scale, color, font_thickness, cv2.LINE_AA)
                    cv2.imwrite(output_filepath, 255 * image)
                render_times_filepath = os.path.join(
                    args.checkpoints_dir, "slimnet_levels_render_times.txt")
            with open(render_times_filepath, "w") as f:
                f.write(str(render_times))
            plot_output_filepath = os.path.join(
                args.checkpoints_dir, "slimnet_levels_render_times.png")
            import matplotlib.pyplot as plt
            plt.plot(1000 * np.array(render_times))
            ax = plt.gca()
            ax.set_ylim([0.8 * 1000 * render_times[0],
                        1.2 * 1000 * render_times[-1]])
            plt.savefig(plot_output_filepath)

    def run_get_train_time(self):
        """Parse the training time from tensorboard logs."""
        from tensorboard.backend.event_processing import event_accumulator
        args = self.args
        logs_dir = os.path.join(args.checkpoints_dir, "logs")
        size_guidance = copy.deepcopy(event_accumulator.DEFAULT_SIZE_GUIDANCE)
        size_guidance[event_accumulator.SCALARS] = 0
        event_acc = event_accumulator.EventAccumulator(
            logs_dir, size_guidance=size_guidance)
        event_acc.Reload()
        loss_events = event_acc.Scalars('train/loss')
        train_time = loss_events[-1].wall_time - \
            event_acc.FirstEventTimestamp()
        print(
            f"Train time: {train_time} seconds or {train_time / 60 / 60} hours")

    def run_benchmark(self):
        """Generates images along the path from the loaded model."""
        args = self.args
        device = self.device
        if args.use_aux_network:
            self.aux_model.half()
        self.model.half()
        resolutions = [
            (self.train_data.original_width // 8,
             self.train_data.original_height // 8),
            (self.train_data.original_width // 4,
             self.train_data.original_height // 4),
            (self.train_data.original_width // 2,
             self.train_data.original_height // 2),
        ]
        if args.script_mode == options.BENCHMARK:
            resolutions += [
                (self.train_data.original_width // 1,
                 self.train_data.original_height // 1)
            ]
        benchmark_results = {f"{width}_{height}": {}
                             for width, height in resolutions}
        batch_size = args.val_batch_size
        if batch_size < 2000000:
            print(f"Warning: Batch size {batch_size} is < 2MP")
        with torch.no_grad():
            for width, height in tqdm.tqdm(resolutions, desc="Resolutions"):
                for lod in tqdm.trange(self.model.num_outputs, desc="LOD", leave=False):
                    self.model.eval()
                    # render_poses = self.train_data.get_render_poses(args.dataset_render_poses)
                    dataset = self.train_data
                    enumerate_rays_times = []
                    nn_inference_times = []
                    full_render_times = []
                    for i in tqdm.trange(2, desc="Burnin", leave=False):
                        if self.train_data.using_processed_camera_parameters:
                            transform = torch.tensor(
                                self.train_data.get_transform(i), device=device).half()
                            intrinsics = torch.tensor(
                                self.train_data.get_intrinsics(i), device=device).half()
                            intrinsics_ratio = min(
                                (height / self.train_data.height), (width / self.train_data.width))
                            intrinsics[:2, :2] = (
                                intrinsics[:2, :2] * intrinsics_ratio)
                            intrinsics[0, 2] = (
                                intrinsics[0, 2] * (width / self.train_data.width))
                            intrinsics[1, 2] = (
                                intrinsics[1, 2] * (height / self.train_data.height))
                            ray_origins, ray_directions = self.train_data.camera_params_to_rays(
                                intrinsics, transform, height, width)
                        else:
                            raise NotImplementedError()
                        if args.model == LEARNED_RAY:
                            rays_plucker = self.model.ray_to_indices(
                                ray_origins, ray_directions)
                        else:
                            rays_plucker = my_torch_utils.convert_rays_to_plucker(
                                ray_origins, ray_directions)
                        rays_with_t = torch.cat(
                            (rays_plucker, 0.0 *
                             torch.ones_like(rays_plucker[:, :, :1])),
                            dim=-1)
                        rendered_image_outputs = self.render_full_image(batch_size,
                                                                        rays_with_t,
                                                                        lods=[
                                                                            lod],
                                                                        aux_discarding=True)
                        assert rendered_image_outputs is not None
                    benchmark_views = range(
                        0, len(self.train_data), args.benchmark_view_step)
                    for frame_num in tqdm.tqdm(benchmark_views, desc="Frame", leave=False):
                        t = 0.0
                        torch.cuda.synchronize()
                        t0 = time.time()
                        if self.train_data.using_processed_camera_parameters:
                            transform = torch.tensor(self.train_data.get_transform(
                                frame_num), device=device).half()
                            intrinsics = torch.tensor(self.train_data.get_intrinsics(
                                frame_num), device=device).half()
                            intrinsics_ratio = min(
                                (height / self.train_data.height), (width / self.train_data.width))
                            intrinsics[:2, :2] = (
                                intrinsics[:2, :2] * intrinsics_ratio)
                            intrinsics[0, 2] = (
                                intrinsics[0, 2] * (width / self.train_data.width))
                            intrinsics[1, 2] = (
                                intrinsics[1, 2] * (height / self.train_data.height))
                            ray_origins, ray_directions = self.train_data.camera_params_to_rays(
                                intrinsics, transform, height, width)
                        else:
                            raise NotImplementedError()
                        if args.model == LEARNED_RAY:
                            rays_plucker = self.model.ray_to_indices(
                                ray_origins, ray_directions)
                        else:
                            rays_plucker = my_torch_utils.convert_rays_to_plucker(
                                ray_origins, ray_directions)
                        rays_with_t = torch.cat(
                            (rays_plucker, t *
                             torch.ones_like(rays_plucker[:, :, :1])),
                            dim=-1)
                        torch.cuda.synchronize()
                        t1 = time.time()
                        rendered_image_outputs = self.render_full_image(batch_size,
                                                                        rays_with_t,
                                                                        lods=[
                                                                            lod],
                                                                        aux_discarding=True)
                        assert rendered_image_outputs is not None
                        torch.cuda.synchronize()
                        t2 = time.time()
                        enumerate_rays_times.append(t1 - t0)
                        nn_inference_times.append(t2 - t1)
                        full_render_times.append(t2 - t0)
                        # if width == 504:
                        #     my_torch_utils.save_torch_image(
                        #         os.path.join(args.checkpoints_dir, "t", f"{frame_num}.png"),
                        #         rendered_image_outputs.full_image
                        #     )
                    mean_enumerate_rays_time = np.mean(enumerate_rays_times)
                    mean_nn_inference_time = np.mean(nn_inference_times)
                    mean_full_render_time = np.mean(full_render_times)
                    benchmark_results[f"{width}_{height}"][lod] = mean_full_render_time
                    benchmark_results[f"{width}_{height}"][f"{lod}_enumerate_rays"] = mean_enumerate_rays_time
                    benchmark_results[f"{width}_{height}"][f"{lod}_nn_inference"] = mean_nn_inference_time
        output_filename = "benchmark.json" if args.script_mode == options.BENCHMARK else "benchmark_slimnet.json"
        output_file_path = os.path.join(args.checkpoints_dir, output_filename)
        with open(output_file_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)

    def run_save_proto(self) -> None:
        """Dumps model weights as a protobuf bin file."""
        model_proto = model_pb2.Model()
        model_proto.primary.CopyFrom(self.model.get_proto())
        if self.args.use_aux_network:
            model_proto.auxiliary.CopyFrom(self.aux_model.get_proto())
        weight_output_file = os.path.join(
            self.args.checkpoints_dir, "weights.bin")
        with open(weight_output_file, "wb") as f:
            f.write(model_proto.SerializeToString())

    def debug_manual_inference(self):
        model_state_dict = self.model.state_dict()
        with torch.no_grad():
            dummy_input = torch.arange(
                self.model.in_features, device=self.device, dtype=torch.float32)
            correct_output = self.model(dummy_input)
            correct_output = dummy_input
            my_output = dummy_input
            for i in range(self.model.layers - 2):
                # Linear
                my_output = (model_state_dict[f"model.{3 * i}.weight"] @ my_output + model_state_dict[
                    f"model.{3 * i}.bias"])
                correct_output = self.model.model[3 * i](correct_output)
                if not torch.allclose(my_output, correct_output, atol=1e-4):
                    print(
                        f"layer {3 * i}, diff {torch.abs(my_output - correct_output).max().item()}")
                    exit(1)
                # LayerNorm
                my_output = (my_output - torch.mean(my_output)) / torch.sqrt(
                    torch.var(my_output, unbiased=False) + 1e-5)
                my_output = my_output * model_state_dict[f"model.{3 * i + 1}.weight"] + model_state_dict[
                    f"model.{3 * i + 1}.bias"]
                correct_output = self.model.model[3 * i + 1](correct_output)
                if not torch.allclose(my_output, correct_output, atol=1e-4):
                    print(
                        f"layer {3 * i + 1}, diff {torch.abs(my_output - correct_output).max().item()}")
                    exit(1)
                # ReLU
                my_output = torch.clamp(my_output, min=0.0)
                correct_output = self.model.model[3 * i + 2](correct_output)
                if not torch.allclose(my_output, correct_output, atol=1e-4):
                    print(
                        f"layer {3 * i + 2}, diff {torch.abs(my_output - correct_output).max().item()}")
                    exit(1)
            my_output = (model_state_dict[f"model.{3 * (self.model.layers - 1)}.weight"] @ my_output +
                         model_state_dict[
                             f"model.{3 * (self.model.layers - 1)}.bias"])
            correct_output = self.model.model[3 *
                                              (self.model.layers - 1)](correct_output)
            print("diff", torch.abs(my_output - correct_output).max())

    def visualize_cameras(self):
        transforms = self.train_data.transforms
        intrinsics = self.train_data.intrinsics
        positions = transforms[:, :3, 3]

        ray_points = []
        ray_points2_colors = []
        for i in range(len(transforms)):
            ray_origin, ray_direction = self.train_data.camera_params_to_rays(
                torch.tensor(intrinsics[i]),
                torch.tensor(transforms[i]),
                height=self.train_data.height,
                width=self.train_data.width
            )
            ray_points.append(ray_origin + 0.1 * ray_direction)
            rr, gg = np.meshgrid(
                np.linspace(0, 1, ray_origin.shape[1]),
                np.linspace(0, 1, ray_origin.shape[0])
            )
            colors = np.stack((rr, gg, np.zeros_like(rr)), axis=-1)
            ray_points2_colors.append(colors)
            if sum(len(x.reshape((-1, 3))) for x in ray_points) > 1e6:
                break
        ray_points = np.stack(ray_points).reshape((-1, 3))
        ray_points2_colors = np.stack(ray_points2_colors).reshape((-1, 3))

        all_pointclouds = []
        camera_positions = o3d.geometry.PointCloud()
        camera_positions.points = o3d.utility.Vector3dVector(positions)
        camera_positions.colors = o3d.utility.Vector3dVector(
            np.ones_like(positions) * np.array((1.0, 0.0, 0.0))[None])
        all_pointclouds.append(camera_positions)
        ray_points_pc = o3d.geometry.PointCloud()
        ray_points_pc.points = o3d.utility.Vector3dVector(ray_points)
        ray_points_pc.colors = o3d.utility.Vector3dVector(ray_points2_colors)
        all_pointclouds.append(ray_points_pc)

        render_poses = self.train_data.get_render_poses(
            self.args.dataset_render_poses)
        positions = render_poses["transforms"][:, :3, 3]

        ray_points2 = []
        ray_points2_colors = []
        for i in range(len(render_poses["transforms"])):
            ray_origin, ray_direction = self.train_data.camera_params_to_rays(
                render_poses["intrinsics"][i],
                render_poses["transforms"][i],
                height=self.train_data.height,
                width=self.train_data.width
            )
            ray_points2.append(ray_origin + 0.1 * ray_direction)
            rr, gg = np.meshgrid(
                np.linspace(0, 1, ray_origin.shape[1]),
                np.linspace(0, 1, ray_origin.shape[0])
            )
            colors = np.stack((rr, gg, np.zeros_like(rr)), axis=-1)
            ray_points2_colors.append(colors)
            if sum(len(x.reshape((-1, 3))) for x in ray_points2) > 1e6:
                break
        ray_points2 = np.stack(ray_points2).reshape((-1, 3))
        ray_points2_colors = np.stack(ray_points2_colors).reshape((-1, 3))

        camera_positions = o3d.geometry.PointCloud()
        camera_positions.points = o3d.utility.Vector3dVector(positions)
        camera_positions.colors = o3d.utility.Vector3dVector(
            np.ones_like(positions) * np.array((1.0, 0.0, 1.0))[None])
        all_pointclouds.append(camera_positions)
        ray_points_pc = o3d.geometry.PointCloud()
        ray_points_pc.points = o3d.utility.Vector3dVector(ray_points2)
        ray_points_pc.colors = o3d.utility.Vector3dVector(ray_points2_colors)
        all_pointclouds.append(ray_points_pc)

        o3d.visualization.draw_geometries(all_pointclouds)

    def run_viewer(self):
        """Starts a viewer."""
        args = self.args
        from utils import viewer_utils
        import glumpy
        width = height = 512
        viewer = viewer_utils.Viewer(width=width, height=height)
        render_poses = self.train_data.get_render_poses(
            args.dataset_render_poses)
        i = 0
        if args.use_aux_network:
            self.aux_model.half()
        self.model.half()

        mouse_pos_x = 0
        mouse_pos_y = 0
        current_lod = 0.0
        target_lod = 0
        last_mouse_press_time = 0
        theta = 0
        phi = 0
        radius = 4.0
        print(f"Using LoD {current_lod}")
        random_map = torch.rand((height, width), device=self.device)
        position_offset = np.zeros(3)

        def draw_callback(dt, h, w):
            t = 0.0
            nonlocal i
            nonlocal current_lod
            lod_step_per_frame = self.model.num_outputs / 1000
            if np.abs(current_lod - target_lod) > lod_step_per_frame:
                current_lod = (current_lod + lod_step_per_frame) % self.model.num_outputs
                print(f"changing LoD to {current_lod}")
            else:
                current_lod = float(target_lod)
            with torch.no_grad():
                intrinsics = render_poses["intrinsics"][i].to(
                    self.device).half()
                intrinsics_ratio = min(
                    (h / self.train_data.height), (w / self.train_data.width))
                intrinsics[:2, :2] = intrinsics[:2, :2] * intrinsics_ratio
                intrinsics[0, 2] = intrinsics[0, 2] * \
                    (w / self.train_data.width)
                intrinsics[1, 2] = intrinsics[1, 2] * \
                    (h / self.train_data.height)

                position = self.train_data.render_poses_target + radius * self.train_data.render_poses_rotation_mat @ np.array(
                    [np.cos(theta), -np.sin(theta), -np.sin(phi)], dtype=np.float16) + position_offset
                rotation = holostudio_reader_utils.look_at(position, self.train_data.render_poses_up,
                                                           self.train_data.render_poses_target + position_offset)
                rotation[:, 0] = -rotation[:, 0]
                rotation[:, 2] = -rotation[:, 2]
                new_transform = np.concatenate(
                    (rotation, np.array((0, 0, 0, 1), dtype=np.float16)[None]), axis=0)

                ray_origins, ray_directions = self.train_data.camera_params_to_rays(
                    intrinsics,
                    torch.tensor(new_transform, device=self.device,
                                 dtype=torch.float16),
                    h, w
                )
                rays_plucker = my_torch_utils.convert_rays_to_plucker(
                    ray_origins, ray_directions)
                rays_with_t = torch.cat(
                    (rays_plucker, t *
                     torch.ones_like(rays_plucker[:, :, :1])),
                    dim=-1)
                i = (i + 1) % args.dataset_render_poses
                if float.is_integer(current_lod):
                    full_image_outputs = self.render_full_image(args.viewer_batch_size,
                                                                rays_with_t,
                                                                lods=[
                                                                    int(current_lod)],
                                                                aux_discarding=True)
                    full_image = full_image_outputs.full_image
                elif args.continuous_lod_training:
                    full_image_outputs = self.render_full_image(args.viewer_batch_size,
                                                                rays_with_t,
                                                                lods=[current_lod],
                                                                aux_discarding=True)
                    full_image = full_image_outputs.full_image
                else:
                    lodmap = int(current_lod) + torch.le(random_map,
                                                         current_lod - np.floor(current_lod)).long()
                    lodmap = torch.remainder(lodmap, self.model.num_outputs)
                    full_image_outputs = self.render_full_image_multilod(args.viewer_batch_size,
                                                                         rays_with_t, lodmap,
                                                                         aux_discarding=True)
                    full_image = full_image_outputs.full_image
                full_image = torch.flipud(torch.clip(full_image, min=0, max=1))
            return full_image.float()

        dragged = False

        def mouse_press_callback(x, y, button):
            nonlocal last_mouse_press_time
            nonlocal dragged
            dragged = False
            last_mouse_press_time = time.time()

        def mouse_release_callback(x, y, button):
            nonlocal current_lod
            nonlocal target_lod
            if not dragged and time.time() - last_mouse_press_time < 0.200:
                # current_lod = ((2 * current_lod + 1) / 2) % self.model.num_outputs
                lod_step = 1 if self.model.num_outputs < 10 else self.model.num_outputs // 5
                target_lod = (target_lod + lod_step) % self.model.num_outputs
                print(f"Using LoD {target_lod}")

        def mouse_drag_callback(x, y, dx, dy, buttons):
            nonlocal mouse_pos_x
            nonlocal mouse_pos_y
            nonlocal position_offset
            nonlocal dragged
            nonlocal theta, phi
            mouse_pos_x = x
            mouse_pos_y = y
            if buttons & 2:
                # Left click down
                theta = np.fmod(theta + 2 * np.pi * dx /
                                width + 2 * np.pi, 2 * np.pi)
                phi = np.clip(phi + np.pi * dy / height, -np.pi / 2, np.pi / 2)
                dragged = True
            elif buttons & 8:
                # Right click down
                old_position_offset = position_offset.copy()
                pan_speed = 0.002
                position = self.train_data.render_poses_target + radius * self.train_data.render_poses_rotation_mat @ np.array(
                    [np.cos(theta), -np.sin(theta), -np.sin(phi)], dtype=np.float16) + old_position_offset
                rotation = holostudio_reader_utils.look_at(position, self.train_data.render_poses_up,
                                                           self.train_data.render_poses_target + old_position_offset)
                rotation[:, 0] = -rotation[:, 0]
                rotation[:, 2] = -rotation[:, 2]
                position_offset += (rotation[:3, :3] @ np.array(
                    [-pan_speed * dx, -pan_speed * dy, 0], dtype=np.float16))
                dragged = True

        def mouse_scroll_callback(x, y, dx, dy):
            nonlocal radius
            radius = np.clip(1.0 - 0.5 * dy, 0.5, 1.5) * radius

        viewer.draw_callback = draw_callback
        viewer.mouse_press_callback = mouse_press_callback
        viewer.mouse_release_callback = mouse_release_callback
        viewer.mouse_drag_callback = mouse_drag_callback
        viewer.mouse_scroll_callback = mouse_scroll_callback
        viewer.run()

    def run_eval(self):
        args = self.args
        device = self.device
        self.model.eval()
        self.aux_model.eval()
        aux_discarding = False
        eval_on_training = args.script_mode in (
            options.EVAL_MEMORIZATION, options.EVAL_MEMORIZATION_MULTIRES)
        eval_multires = args.script_mode in (
            options.EVAL_MULTIRES, options.EVAL_MEMORIZATION_MULTIRES)
        res_factors = [1, 2, 4, 8] if eval_multires else [1]
        if eval_on_training:
            datasets = {"train": self.train_data}
        else:
            datasets = {"val_dataset": self.val_data,
                        "test_dataset": self.test_data}
        eval_lods = range(self.model.num_outputs)
        if args.script_mode == options.EVAL_SLIMNET_LODS:
            scales = [1]
            eval_lods = range(self.model.num_outputs)
            flicker_per_lod = [[] for _ in range(self.model.num_outputs)]
            aux_discarding = True
        elif args.model == SLIMNET:
            scales = [8, 4, 2, 1]
            eval_lods = [self.model.scale_to_lod(x) for x in scales]
        num_params = [self.model.num_params(lod) for lod in eval_lods]
        for dataset_name, dataset in datasets.items():
            if dataset is None:
                continue
            for factor in tqdm.tqdm(res_factors, desc="Factors"):
                dataset.update_factor(factor)
                data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                         num_workers=args.dataloader_num_workers)
                masked_psnr_values = [[] for _ in range(len(eval_lods))]
                psnr_values = [[] for _ in range(len(eval_lods))]
                ssim_values = [[] for _ in range(len(eval_lods))]
                render_times = [[] for _ in range(len(eval_lods))]
                cropped_psnr_values = [[] for _ in range(len(eval_lods))]
                cropped_ssim_values = [[] for _ in range(len(eval_lods))]
                with torch.no_grad():
                    for frame_num, data in enumerate(tqdm.tqdm(data_loader, desc="Frame Batch", leave=False)):
                        frame_batch_size, height, width, data_image_channels = data["image"].shape
                        ray_colors = data["image"].to(device)
                        ray_masks = data["mask"].to(
                            device) if "mask" in data else None
                        ray_t = (torch.zeros_like(
                            data["image"][:, :, :, 0]) + data["t"][:, None, None]).to(device)
                        ray_origins = []
                        ray_directions = []
                        for i in range(frame_batch_size):
                            if self.train_data.using_processed_camera_parameters:
                                transform = data["transform"][i].to(device)
                                intrinsics = data["intrinsics"][i].to(device)
                                frame_ray_origins, frame_ray_directions = self.train_data.camera_params_to_rays(
                                    intrinsics, transform, height, width)
                            else:
                                pose_target = data["pose"][0, :3, :4]
                                focal = self.train_data.focal
                                frame_ray_origins, frame_ray_directions = nerf_utils.get_ray_bundle(height, width,
                                                                                                    focal,
                                                                                                    pose_target)
                            ray_origins.append(frame_ray_origins)
                            ray_directions.append(frame_ray_directions)
                        ray_origins = torch.cat(ray_origins)
                        ray_directions = torch.cat(ray_directions)
                        if args.model == LEARNED_RAY:
                            rays_plucker = self.model.ray_to_indices(
                                ray_origins, ray_directions)
                        else:
                            rays_plucker = my_torch_utils.convert_rays_to_plucker(
                                ray_origins, ray_directions)
                        rays_with_t = torch.cat(
                            (rays_plucker, ray_t[0, :, :, None]), dim=-1)
                        reference = data["image"][0, :, :, :3].float().to(device)
                        prev_frame = None
                        for i, lod in enumerate(tqdm.tqdm(eval_lods, desc="LoDs", leave=False)):
                            t0 = time.time()
                            rendered_image_outputs = self.render_full_image(
                                args.val_batch_size, rays_with_t, early_stopping=True, lods=[lod],
                                aux_discarding=aux_discarding)
                            t1 = time.time()
                            render_time_seconds = t1 - t0
                            psnr_val = pytorch_psnr.psnr(
                                ray_colors[:1, :, :, :3].permute((0, 3, 1, 2)),
                                rendered_image_outputs.full_image[None, :, :, :3].permute(
                                    (0, 3, 1, 2))
                            ).cpu().numpy().astype(float)
                            masked_psnr_val = pytorch_psnr.psnr(
                                ray_colors[:1, :, :, :3].permute((0, 3, 1, 2)),
                                rendered_image_outputs.full_image[None, :, :, :3].permute(
                                    (0, 3, 1, 2)),
                                mask=ray_masks[:, None, :, :]
                            ).cpu().numpy().astype(float)
                            ssim_val = pytorch_ssim.ssim(
                                ray_colors[:, :, :, :3].permute((0, 3, 1, 2)),
                                rendered_image_outputs.full_image[None, :, :, :3].permute(
                                    (0, 3, 1, 2))).cpu().numpy().astype(float)
                            cropped_psnr_val = pytorch_psnr.cropped_psnr(
                                ray_colors[:1, :, :, :].permute((0, 3, 1, 2)),
                                rendered_image_outputs.full_image[None, :, :, :].permute(
                                    (0, 3, 1, 2))
                            )
                            cropped_ssim_val = pytorch_ssim.cropped_ssim(
                                ray_colors[:1, :, :, :].permute((0, 3, 1, 2)),
                                rendered_image_outputs.full_image[None, :, :, :].permute(
                                    (0, 3, 1, 2))
                            )
                            masked_psnr_values[i].append(
                                float(masked_psnr_val))
                            psnr_values[i].append(float(psnr_val))
                            ssim_values[i].append(float(ssim_val))
                            render_times[i].append(
                                float(render_time_seconds))
                            cropped_psnr_values[i].append(
                                float(cropped_psnr_val))
                            cropped_ssim_values[i].append(
                                float(cropped_ssim_val))
                            if args.script_mode == options.EVAL_SLIMNET_LODS:
                                current_frame = torch.clamp(
                                    rendered_image_outputs.full_image[:, :, :3], 0, 1).float()
                                if i > 0:
                                    flicker = fft_flicker.compute_fft_flicker_torch(
                                        prev_frame, reference, current_frame, reference)
                                    flicker_per_lod[lod].append(flicker.item())
                                prev_frame = current_frame
                avg_masked_psnr_values = [
                    np.mean(x) for x in masked_psnr_values]
                avg_psnr_values = [np.mean(x) for x in psnr_values]
                avg_ssim_values = [np.mean(x) for x in ssim_values]
                avg_render_times = [np.mean(x) for x in render_times]
                avg_cropped_psnr_values = [
                    np.mean(x) for x in cropped_psnr_values]
                avg_cropped_ssim_values = [
                    np.mean(x) for x in cropped_ssim_values]
                weights_size_mbytes = [
                    x * 4 / (1024 * 1024) for x in num_params]
                output_filename = f"eval_{dataset_name}_{factor}.json"
                if args.script_mode == options.EVAL_SLIMNET_LODS:
                    output_filename = f"eval_slimnet_{dataset_name}_{factor}.json"
                output_file_path = os.path.join(
                    args.checkpoints_dir, output_filename)
                with open(output_file_path, "w") as f:
                    json.dump({
                        "avg_masked_psnr_values": avg_masked_psnr_values,
                        "avg_psnr_values": avg_psnr_values,
                        "avg_ssim_values": avg_ssim_values,
                        "avg_render_times": avg_render_times,
                        "avg_cropped_psnr_values": avg_cropped_psnr_values,
                        "avg_cropped_ssim_values": avg_cropped_ssim_values,
                        "num_params": num_params,
                        "weights_size_mbytes": weights_size_mbytes
                    }, f, indent=2)
                if args.script_mode != options.EVAL_SLIMNET_LODS:
                    output_file_path2 = os.path.join(
                        args.checkpoints_dir, f"eval_{dataset_name}_{factor}_full.json")
                    with open(output_file_path2, "w") as f:
                        json.dump({
                            "masked_psnr_values": masked_psnr_values,
                            "psnr_values": psnr_values,
                            "ssim_values": ssim_values,
                            "render_times": render_times,
                            "cropped_psnr_values": cropped_psnr_values,
                            "cropped_ssim_values": cropped_ssim_values
                        }, f, indent=2)
                if args.script_mode == options.EVAL_SLIMNET_LODS:
                    # Output flicker results
                    avg_flicker_per_lod = [(np.mean(x) if x else 0)
                                           for x in flicker_per_lod]
                    output_filename = f"eval_flicker_{dataset_name}.json"
                    output_file_path = os.path.join(
                        args.checkpoints_dir, output_filename)
                    with open(output_file_path, "w") as f:
                        json.dump(avg_flicker_per_lod, f, indent=2)

    def run_eval_flicker(self):
        """Evaluate the popping effect from switching LODs"""
        args = self.args
        device = self.device
        eval_with_half = True
        self.model.eval()
        if eval_with_half:
            self.model.half()
        if self.aux_model:
            self.aux_model.eval()
            if eval_with_half:
                self.aux_model.half()
        flicker_dither_lods = 385
        datasets = {"val_dataset": self.val_data,
                    "test_dataset": self.test_data}
        with torch.no_grad():
            for dataset_name, dataset in datasets.items():
                print(f"Evaluating flicker for {dataset_name}")
                random_map = torch.rand((dataset.height, dataset.width), device=self.device)
                flicker_per_lod = [[] for _ in range(self.model.num_outputs)]
                if args.script_mode == options.EVAL_FLICKER_DITHER:
                    flicker_per_lod = [[] for _ in range(flicker_dither_lods)]
                data_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                         num_workers=args.dataloader_num_workers)
                for frame_num, data in enumerate(tqdm.tqdm(data_loader, desc="Frame Batch", leave=False)):
                    frame_batch_size, height, width, data_image_channels = data["image"].shape
                    ray_colors = data["image"].to(device)
                    ray_t = (torch.zeros_like(
                        data["image"][:, :, :, 0]) + data["t"][:, None, None]).to(device)
                    ray_origins = []
                    ray_directions = []
                    for i in range(frame_batch_size):
                        if self.train_data.using_processed_camera_parameters:
                            transform = data["transform"][i].to(device)
                            intrinsics = data["intrinsics"][i].to(device)
                            frame_ray_origins, frame_ray_directions = self.train_data.camera_params_to_rays(
                                intrinsics, transform, height, width)
                        else:
                            pose_target = data["pose"][0, :3, :4]
                            focal = self.train_data.focal
                            frame_ray_origins, frame_ray_directions = nerf_utils.get_ray_bundle(height, width,
                                                                                                focal,
                                                                                                pose_target)
                        ray_origins.append(frame_ray_origins)
                        ray_directions.append(frame_ray_directions)
                    ray_origins = torch.cat(ray_origins)
                    ray_directions = torch.cat(ray_directions)
                    if args.model == LEARNED_RAY:
                        rays_plucker = self.model.ray_to_indices(
                            ray_origins, ray_directions)
                    else:
                        rays_plucker = my_torch_utils.convert_rays_to_plucker(
                            ray_origins, ray_directions)
                    rays_with_t = torch.cat(
                        (rays_plucker, ray_t[0, :, :, None]), dim=-1)
                    if eval_with_half:
                        rays_with_t = rays_with_t.half()
                    lods = tqdm.trange(
                        0, self.model.num_outputs, desc="LOD", leave=False)
                    if args.script_mode == options.EVAL_FLICKER_DITHER:
                        lods = tqdm.tqdm(
                            np.linspace(0, self.model.num_outputs - 1, flicker_dither_lods),
                            desc="LOD", leave=False
                        )
                    reference = ray_colors[0, :, :, :3].float()
                    prev_frame = None
                    for i, lod in enumerate(lods):
                        if args.script_mode == options.EVAL_FLICKER_DITHER:
                            lodmap = int(lod) + torch.le(random_map, lod - np.floor(lod)).long()
                            lodmap = torch.remainder(lodmap, self.model.num_outputs)
                            rendered_image_outputs = self.render_full_image_multilod(
                                args.val_batch_size, rays_with_t, lodmap, aux_discarding=True)
                        else:
                            rendered_image_outputs = self.render_full_image(
                                args.val_batch_size, rays_with_t, early_stopping=True,
                                aux_discarding=True, lods=[lod])
                        current_frame = torch.clamp(
                            rendered_image_outputs.full_image[:, :, :3], 0, 1).float()
                        if i > 0:
                            flicker = fft_flicker.compute_fft_flicker_torch(
                                prev_frame, reference, current_frame, reference)
                            flicker_per_lod[i].append(flicker.item())
                        prev_frame = current_frame
                avg_flicker_per_lod = [(np.mean(x) if x else 0)
                                       for x in flicker_per_lod]
                output_filename = f"eval_flicker_{dataset_name}.json"
                if args.script_mode == options.EVAL_FLICKER_DITHER:
                    output_filename = f"eval_flicker_dither_{dataset_name}.json"
                output_file_path = os.path.join(
                    args.checkpoints_dir, output_filename)
                with open(output_file_path, "w") as f:
                    json.dump(avg_flicker_per_lod, f, indent=2)

    def run_eval_update_modelsize(self):
        "Fix the model size in eval files"
        args = self.args
        eval_files = glob.glob(os.path.join(
            args.checkpoints_dir, "eval_*.json"))

        scales = [8, 4, 2, 1]
        eval_lods = [self.model.scale_to_lod(x) for x in scales]
        num_params = [self.model.num_params(lod) for lod in eval_lods]
        weights_size_mbytes = [x * 4 / (1024 * 1024) for x in num_params]
        num_params_slimnet = [self.model.num_params(
            lod) for lod in range(self.model.num_outputs)]
        weights_size_mbytes_slimnet = [
            x * 4 / (1024 * 1024) for x in num_params_slimnet]
        for eval_file in eval_files:
            with open(eval_file, "r") as f:
                eval_data = json.load(f)
            if "weights_size_mbytes" in eval_data:
                if "eval_slimnet" in eval_file:
                    eval_data["num_params"] = num_params
                    eval_data["weights_size_mbytes"] = weights_size_mbytes_slimnet
                else:
                    eval_data["num_params"] = num_params_slimnet
                    eval_data["weights_size_mbytes"] = weights_size_mbytes
                with open(eval_file, "w") as f:
                    json.dump(eval_data, f, indent=2)

    def run_render_occupancy_map(self):
        """Renders the occupancy map."""
        args = self.args
        occupancy_map_dir = my_utils.join_and_make(
            args.checkpoints_dir, "occupancy_map")
        occupancy_map_rgba_dir = my_utils.join_and_make(
            args.checkpoints_dir, "occupancy_map_rgba")
        device = self.device
        self.model.eval()
        self.model.half()
        if self.aux_model:
            self.aux_model.half()
        with torch.no_grad():
            self.model.eval()
            batch_size = 2000000
            num_render_poses = 30
            render_poses = self.train_data.get_render_poses(num_render_poses)
            for i in tqdm.tqdm(range(num_render_poses), desc="Inference"):
                t = (self.train_data.num_frames - 1) * \
                    i / (args.dataset_render_poses - 1)
                height = self.train_data.height
                width = self.train_data.width
                if self.train_data.using_processed_camera_parameters:
                    ray_origins, ray_directions = self.train_data.camera_params_to_rays(
                        render_poses["intrinsics"][i].to(self.device).half(),
                        render_poses["transforms"][i].to(self.device).half(),
                        height, width
                    )
                else:
                    focal = self.train_data.focal
                    pose = render_poses[i]
                    ray_origins, ray_directions = nerf_utils.get_ray_bundle(height, width, focal,
                                                                            pose[:3, :4].to(device).half())
                if args.model == LEARNED_RAY:
                    rays_plucker = self.model.ray_to_indices(
                        ray_origins, ray_directions)
                else:
                    rays_plucker = my_torch_utils.convert_rays_to_plucker(
                        ray_origins, ray_directions)
                rays_with_t = torch.cat(
                    (rays_plucker, t *
                     torch.ones_like(rays_plucker[:, :, :1])),
                    dim=-1)
                rendered_image_outputs = self.render_full_image(batch_size,
                                                                rays_with_t,
                                                                early_stopping=True,
                                                                aux_discarding=True)
                crop_h = (1300 + np.array((0000, 2000))) / \
                    self.args.dataset_resize_factor
                crop_w = (1050 + np.array((0000, 2000))) / \
                    self.args.dataset_resize_factor
                crop_h = np.round(crop_h).astype(np.long)
                crop_w = np.round(crop_w).astype(np.long)
                rgb_image = rendered_image_outputs.layer_outputs[-2]
                my_torch_utils.save_torch_image(
                    os.path.join(occupancy_map_rgba_dir, "%05d.png" % i),
                    rgb_image[crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]
                )
                occupancy_map_image = torch.ones(
                    (height, width, 3), dtype=torch.float32, device=device)
                occupancy_map_image[rendered_image_outputs.aux_image[:,
                                                                     :, 0] > args.aux_discard_threshold] = 0
                occupancy_map_image = occupancy_map_image[crop_h[0]:crop_h[1], crop_w[0]:crop_w[1]]
                my_torch_utils.save_torch_image(
                    os.path.join(occupancy_map_dir, "%05d.png" % i),
                    occupancy_map_image
                )

    def run_render_transition(self):
        """Renders a single pose with multiple levels of detail."""
        args = self.args
        device = self.device
        output_dir = args.checkpoints_dir
        output_frames_dir = os.path.join(output_dir, "transition_frames")
        my_utils.rmdir_sync(output_frames_dir)
        os.makedirs(output_frames_dir, exist_ok=True)
        start_lod = 0
        max_lod = self.model.num_outputs - 1
        num_frames = 400
        if args.model == MIPNET:
            num_frames = 4
        height = self.train_data.height
        width = self.train_data.width
        if "StudioV2CaptureTest0805_max_light" in args.dataset_path:
            poses = [10]
            crops = np.array([(1645, 1446, 2310, 2899)])
        elif "DavidDataset0113Config4v3" in args.dataset_path:
            poses = [55]
            crops = np.array([(1637, 1232, 2365, 2821)])
        elif "MariaCamera0124Take2" in args.dataset_path:
            poses = [31]
            crops = np.array([(1843, 1500, 2373, 2963)])
        elif "DavidDataset0113Config5v4" in args.dataset_path:
            poses = [14]
            crops = np.array([(1659, 1450, 2278, 2945)])
        elif "sida_simba_take1v3" in args.dataset_path:
            poses = [0]
            crops = np.array([(1671, 1988, 2762, 3008)])
        elif "Sharon0505_05s" in args.dataset_path:
            poses = [39]
            crops = np.array([(1565, 1510, 2306, 3008)])
        elif "Sharon0505" in args.dataset_path:
            poses = [69]
            crops = np.array([(1540, 1387, 2348, 2907)])
        elif "BaltimoreFoodPrep0218Take2" in args.dataset_path:
            poses = [28]
            crops = np.array([(1635, 1282, 2481, 2983)])
        else:
            print(f"Unknown dataset: {args.dataset_path}")
            exit(1)
        pose_num = poses[0]
        crop = crops[0]
        random_map = torch.rand((height, width), device=self.device)

        if args.model not in ADAPTIVE_NETWORKS:
            raise ValueError("Model is not adaptive")

        with torch.no_grad():
            self.model.eval()
            batch_size = args.val_batch_size
            render_poses = self.train_data.get_render_poses(
                args.dataset_render_poses)
            for i in tqdm.tqdm(range(num_frames), desc="Rendering"):
                current_lod = (start_lod + 
                    (max_lod - start_lod) * (i / (num_frames - 1)))
                if np.abs(current_lod - np.round(current_lod)) < 1e-2:
                    current_lod = np.round(current_lod)
                t = 0
                if self.train_data.using_processed_camera_parameters:
                    ray_origins, ray_directions = self.train_data.camera_params_to_rays(
                        render_poses["intrinsics"][pose_num].to(self.device),
                        render_poses["transforms"][pose_num].to(self.device),
                        height, width
                    )
                else:
                    focal = self.train_data.focal
                    pose = render_poses[pose_num]
                    ray_origins, ray_directions = nerf_utils.get_ray_bundle(height, width, focal,
                                                                            pose[:3, :4].to(device))
                if args.model == LEARNED_RAY:
                    rays_plucker = self.model.ray_to_indices(
                        ray_origins, ray_directions)
                else:
                    rays_plucker = my_torch_utils.convert_rays_to_plucker(
                        ray_origins, ray_directions)
                rays_with_t = torch.cat(
                    (rays_plucker, t *
                     torch.ones_like(rays_plucker[:, :, :1])),
                    dim=-1)
                # Apply crop to rays
                rays_with_t = rays_with_t[crop[1]:crop[3], crop[0]:crop[2]]
                if args.continuous_lod_training or float.is_integer(current_lod):
                    full_image_outputs = self.render_full_image(batch_size,
                                                                rays_with_t,
                                                                lods=[
                                                                    int(current_lod)],
                                                                aux_discarding=True)
                    full_image = full_image_outputs.full_image
                else:
                    lodmap = int(current_lod) + torch.le(random_map,
                                                         current_lod - np.floor(current_lod)).long()
                    lodmap = torch.remainder(lodmap, self.model.num_outputs)
                    full_image_outputs = self.render_full_image_multilod(batch_size,
                                                                         rays_with_t, lodmap,
                                                                         aux_discarding=True)
                    full_image = full_image_outputs.full_image
                full_image_cpu = full_image.cpu().numpy()
                # Add LOD text to image
                full_image_cpu = my_utils.append_text_to_image(
                    full_image_cpu, f"LOD: {current_lod+1:.02f}",
                    font_size=1, font_thickness=5, font_color=(0,0,0,1)
                )
                # Save image
                image_path = os.path.join(output_frames_dir, f"{i:04d}.png")
                full_image_cpu = cv2.cvtColor(full_image_cpu, cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(image_path, full_image_cpu * 255)
                
    def run_render_foveation(self):
        """Renders a single pose with multiple levels of detail."""
        args = self.args
        device = self.device
        output_dir = args.checkpoints_dir
        output_frames_dir = os.path.join(output_dir, "foveation_frames")
        # my_utils.rmdir_sync(output_frames_dir)
        os.makedirs(output_frames_dir, exist_ok=True)
        height = self.train_data.height
        width = self.train_data.width
        random_map = torch.rand((height, width), device=self.device)
        max_lod = self.model.num_outputs - 1

        if "DavidDataset0113Config4" in self.args.dataset_path:
            fovea_position = np.array(
                [2025, 1892]) / self.args.dataset_resize_factor
            pose_num = 46
        elif "DavidDataset0113Config5" in self.args.dataset_path:
            fovea_position = np.array(
                [2025, 1892]) / self.args.dataset_resize_factor
            pose_num = 40
        elif "StudioV2CaptureTest0805_max_light" in self.args.dataset_path:
            fovea_position = np.array(
                [2165, 1560]) / self.args.dataset_resize_factor
            pose_num = 16
        else:
            raise NotImplementedError("Unknown run dataset")

        fovea_position = torch.tensor(
            fovea_position, dtype=torch.float32, device=device)
        xx, yy = torch.meshgrid(torch.arange(width, dtype=torch.float32, device=device),
                                torch.arange(
                                    height, dtype=torch.float32, device=device),
                                indexing="xy")
        pixel_coordinates = torch.stack((xx, yy), dim=2)
        distances = pixel_coordinates - fovea_position
        distances = torch.linalg.norm(
            distances, dim=2) * self.args.dataset_resize_factor
        # distances2 = 0.8 * torch.log(1.0 + distances)
        distances2 = 0.8 * torch.pow(distances, 0.30)
        distances2_floor = torch.floor(distances2)
        distances2_frac = distances2 - distances2_floor
        distances2_frac = torch.sigmoid(100000.0 * (distances2_frac - 0.5))
        distances2 = distances2_floor + distances2_frac
        if args.model in (options.SLIMNET, options.LCSLIMNET):
            lod = 4 - (0.5 * torch.pow(distances, 0.30))
            lod_width = 128 * lod
            lod = lod_width - self.model.min_nodes
            # lod = 20 * torch.round(lod/20)
            lod = torch.clip(lod, min=128-self.model.min_nodes, max=max_lod)
            print("lod min", lod.min(), lod.max())
            print("number of LODs", torch.unique(lod).numel())
        else:
            lod = torch.clip(max_lod + 1 - distances2, min=0, max=max_lod)
        lod_floored = torch.floor(lod)
        lod_frac = lod - lod_floored

        if args.model not in ADAPTIVE_NETWORKS:
            raise ValueError("Model is not adaptive")

        with torch.no_grad():
            self.model.eval()
            batch_size = args.val_batch_size
            render_poses = self.train_data.get_render_poses(
                args.dataset_render_poses)

            t = 0
            if self.train_data.using_processed_camera_parameters:
                ray_origins, ray_directions = self.train_data.camera_params_to_rays(
                    render_poses["intrinsics"][pose_num].to(self.device),
                    render_poses["transforms"][pose_num].to(self.device),
                    height, width
                )
            else:
                focal = self.train_data.focal
                pose = render_poses[pose_num]
                ray_origins, ray_directions = nerf_utils.get_ray_bundle(height, width, focal,
                                                                        pose[:3, :4].to(device))
            if args.model == LEARNED_RAY:
                rays_plucker = self.model.ray_to_indices(
                    ray_origins, ray_directions)
            else:
                rays_plucker = my_torch_utils.convert_rays_to_plucker(
                    ray_origins, ray_directions)
            rays_with_t = torch.cat(
                (rays_plucker, t * torch.ones_like(rays_plucker[:, :, :1])),
                dim=-1)
            lodmap = (lod_floored + torch.le(random_map, lod_frac)).long()
            lodmap = torch.remainder(lodmap, self.model.num_outputs)
            full_image_outputs = self.render_full_image_multilod(batch_size,
                                                                 rays_with_t, lodmap,
                                                                 aux_discarding=True)
            full_image = full_image_outputs.full_image
            my_torch_utils.save_torch_image(
                os.path.join(output_frames_dir, f"render.png"),
                full_image
            )

            print("lodmap.shape", lodmap.shape, lodmap.min(), lodmap.max())
            my_torch_utils.save_torch_image(
                os.path.join(output_frames_dir, f"lodmap.png"),
                lodmap,
                cmap="gray",
                clamp=False,
                vmin=lodmap.min(),
                vmax=lodmap.max()
            )

            lodmap = max_lod * torch.ones_like(lodmap)
            full_image_outputs = self.render_full_image_multilod(batch_size,
                                                                 rays_with_t, lodmap,
                                                                 aux_discarding=True)
            full_image = full_image_outputs.full_image
            my_torch_utils.save_torch_image(
                os.path.join(output_frames_dir, f"unfoveated.png"),
                full_image
            )

    def run_render_slimnet_teaser(self):
        """Render slimnet teaser images."""
        args = self.args
        device = self.device

        self.model.eval()
        batch_size = args.val_batch_size

        with torch.no_grad():
            assert args.dataset_render_poses == 120, "Render poses must be 120"
            render_poses = self.train_data.get_render_poses(
                args.dataset_render_poses)
            if self.model.num_outputs == 462:
                lods = [0, 114, 229, 344, 461]
            elif self.model.num_outputs == 385:
                lods = [0, 76, 153, 230, 384]
            else:
                raise NotImplementedError("Unknown number of outputs")
            if "MariaCamera0124Take2" in args.dataset_path:
                # lods = [0, 76, 153, 230, 384]
                # poses = np.arange(47, 47-5*5, -5)
                # crops = np.array([(1680, 860, 2300, 2400)]).repeat(5, axis=0)
                lods = [0, 76, 153, 230, 384, 230, 153, 76, 0]
                poses = np.arange(37 + 4*5, 37-4*5-1, -5)
                assert len(lods) == len(poses)
                crops = np.array([(1680, 860, 2300, 2400)]).repeat(len(lods), axis=0)
            else:
                exit(1)
            lod_pose = zip(lods, poses)
            for i, (lod, pose_num) in enumerate(tqdm.tqdm(lod_pose, desc="Pose", leave=False)):
                t = ((self.train_data.num_frames - 1) *
                     pose_num / (args.dataset_render_poses - 1))
                height = self.train_data.height
                width = self.train_data.width
                if self.train_data.using_processed_camera_parameters:
                    intrinsics = render_poses["intrinsics"][pose_num].to(
                        self.device)
                    transforms = render_poses["transforms"][pose_num].to(
                        self.device)
                    if args.inference_halves:
                        intrinsics = intrinsics.half()
                        transforms = transforms.half()
                    ray_origins, ray_directions = self.train_data.camera_params_to_rays(
                        intrinsics,
                        transforms,
                        height, width
                    )
                else:
                    focal = self.train_data.focal
                    pose = render_poses[pose_num]
                    ray_origins, ray_directions = nerf_utils.get_ray_bundle(height, width, focal,
                                                                            pose[:3, :4].to(device))
                if args.model == LEARNED_RAY:
                    rays_plucker = self.model.ray_to_indices(
                        ray_origins, ray_directions)
                else:
                    rays_plucker = my_torch_utils.convert_rays_to_plucker(
                        ray_origins, ray_directions)
                rays_with_t = torch.cat(
                    (rays_plucker, t *
                        torch.ones_like(rays_plucker[:, :, :1])),
                    dim=-1)
                my_crop = crops[i]
                rays_with_t = rays_with_t[my_crop[1]
                    :my_crop[3], my_crop[0]:my_crop[2]]
                rendered_image_outputs = self.render_full_image(batch_size,
                                                                rays_with_t,
                                                                lods=[lod],
                                                                early_stopping=True,
                                                                aux_discarding=True)
                image = rendered_image_outputs.full_image.cpu().numpy()
                output_frames_dir = my_utils.join_and_make(
                    args.checkpoints_dir, "teaser_slimnet")
                output_filepath = os.path.join(output_frames_dir,
                                               f"view{pose_num}_lod{lod}.png")
                font = cv2.FONT_HERSHEY_SIMPLEX
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
                cv2.imwrite(output_filepath, 255 * image)

    def run_render_slimnet_loop(self):
        """Generates images along the path from the loaded model."""
        args = self.args
        device = self.device
        output_dir = args.checkpoints_dir
        output_frames_dir = os.path.join(output_dir, "slimnet_loop", "frames")
        os.makedirs(output_frames_dir, exist_ok=True)
        ffmpeg_bin = shutil.which("ffmpeg")
        if args.inference_halves:
            self.model.half()
            self.aux_model.half()
            print("using half precision")
        burned_in = False
        if args.model != SLIMNET:
            raise NotImplementedError("Only slimnet is supported")

        num_frames = 400

        with torch.no_grad():
            self.model.eval()
            batch_size = args.val_batch_size
            render_poses = self.train_data.get_render_poses(num_frames)
            frame_num = 0
            def render_loop(render_poses, lods):
                nonlocal frame_num
                for i in tqdm.tqdm(range(len(lods)), desc="Inference"):
                    t = (self.train_data.num_frames - 1) * \
                        i / (args.dataset_render_poses - 1)
                    height = self.train_data.height
                    width = self.train_data.width
                    if self.train_data.using_processed_camera_parameters:
                        intrinsics = render_poses["intrinsics"][i].to(self.device)
                        transforms = render_poses["transforms"][i].to(self.device)
                        if args.inference_halves:
                            intrinsics = intrinsics.half()
                            transforms = transforms.half()
                        ray_origins, ray_directions = self.train_data.camera_params_to_rays(
                            intrinsics,
                            transforms,
                            height, width
                        )
                    else:
                        focal = self.train_data.focal
                        pose = render_poses[i]
                        ray_origins, ray_directions = nerf_utils.get_ray_bundle(height, width, focal,
                                                                                pose[:3, :4].to(device))
                    if args.model == LEARNED_RAY:
                        rays_plucker = self.model.ray_to_indices(
                            ray_origins, ray_directions)
                    else:
                        rays_plucker = my_torch_utils.convert_rays_to_plucker(
                            ray_origins, ray_directions)
                    rays_with_t = torch.cat(
                        (rays_plucker, t *
                        torch.ones_like(rays_plucker[:, :, :1])),
                        dim=-1)
                    rendered_image_outputs = self.render_full_image(batch_size,
                                                                    rays_with_t,
                                                                    lods=[lods[i]],
                                                                    early_stopping=True,
                                                                    aux_discarding=True)
                    image = rendered_image_outputs.full_image.cpu().numpy()
                    save_path = os.path.join(
                        output_frames_dir, f"{frame_num:05d}.png")
                    image_with_number = my_utils.append_text_to_image(image, f"LOD {lods[i] + 1:.02f}",
                                                                      font_size=1, font_color=(0, 0, 0, 1),
                                                                      font_thickness=5)
                    image_with_number = cv2.cvtColor(
                        image_with_number, cv2.COLOR_RGBA2BGRA)
                    cv2.imwrite(save_path, 255 * image_with_number)
                    frame_num += 1
            render_loop(render_poses, [self.model.num_outputs - 1] * len(render_poses["transforms"]))
            if args.continuous_lod_training:
                render_loop(render_poses, np.linspace(self.model.num_outputs - 1, 0, num_frames))
                render_loop(render_poses, np.linspace(0, self.model.num_outputs - 1, num_frames))
            else:
                render_loop(render_poses, np.arange(self.model.num_outputs - 1, -1, -1))
                render_loop(render_poses, np.arange(self.model.num_outputs))

    def output_saliency_sampling_illustration(self):
        args = self.args
        device = self.device

        if "StudioV2CaptureTest0805_max_light" in args.dataset_path:
            data = self.train_data[103]
        else:
            data = self.train_data[0]

        data_image = data["image"].to(device)
        height, width, _ = data["image"].shape
        print(f"height: {height}, width: {width}")

        mask_image = torch.tensor(data["mask"])
        saliency_image = torch.tensor(data["saliency"])
        sample_probability_image = my_utils.lerp(
            1, saliency_image, args.saliency_sampling_factor_v2)
        sample_probability_image = mask_image * sample_probability_image

        crop = (1430, 777, 2801, 2100)
        data_image = data_image[crop[1]:crop[3], crop[0]:crop[2]]
        mask_image = mask_image[crop[1]:crop[3], crop[0]:crop[2]]
        saliency_image = saliency_image[crop[1]:crop[3], crop[0]:crop[2]]
        sample_probability_image = sample_probability_image[crop[1]
            :crop[3], crop[0]:crop[2]]

        my_torch_utils.save_torch_image("temp/rgba.png", data_image)
        my_torch_utils.save_torch_image(
            "temp/mask.png", mask_image, cmap="gray")
        my_torch_utils.save_torch_image("temp/saliency.png", saliency_image)
        my_torch_utils.save_torch_image(
            "temp/probabilities.png", sample_probability_image, vmin=0, vmax=1.0)

    def run_render_neuron_masking_example(self):
        "Render an example of using neuron masking"
        """Method to render all levels of details from slimnet."""
        args = self.args
        device = self.device

        self.model.eval()
        batch_size = args.val_batch_size

        if args.use_aux_network:
            self.aux_model.half()
        self.model.half()

        assert args.model == SLIMNET and args.lcslimnet_lodmasking, "This method only works with continuous lod slimnet"

        output_frames_dir = os.path.join(
            args.checkpoints_dir, "neuron_masking_example")
        os.makedirs(output_frames_dir, exist_ok=True)

        with torch.no_grad():
            render_times = []
            render_poses = self.train_data.get_render_poses(
                args.dataset_render_poses)
            # Shows the leg of the cart expanding
            lods = np.linspace(198, 199, 90)
            for lod_index, lod in enumerate(lods):
                if "StudioV2CaptureTest0805_max_light" in args.dataset_path:
                    poses = [10]
                    crops = np.array([(1645, 1446, 2310, 2899)])
                elif "DavidDataset0113Config4v3" in args.dataset_path:
                    poses = [55]
                    crops = np.array([(1637, 1232, 2365, 2821)])
                elif "MariaCamera0124Take2" in args.dataset_path:
                    poses = [31]
                    crops = np.array([(1843, 1500, 2373, 2963)])
                elif "DavidDataset0113Config5v4" in args.dataset_path:
                    poses = [14]
                    crops = np.array([(1659, 1450, 2278, 2945)])
                elif "sida_simba_take1v3" in args.dataset_path:
                    poses = [0]
                    crops = np.array([(1671, 1988, 2762, 3008)])
                elif "Sharon0505_05s" in args.dataset_path:
                    poses = [39]
                    crops = np.array([(1565, 1510, 2306, 3008)])
                elif "Sharon0505" in args.dataset_path:
                    poses = [69]
                    crops = np.array([(1540, 1387, 2348, 2907)])
                elif "BaltimoreFoodPrep0218Take2" in args.dataset_path:
                    poses = [28]
                    crops = np.array([(1635, 1282, 2481, 2983)])
                else:
                    print(f"Unknown dataset: {args.dataset_path}")
                    exit(1)
                if args.dataset_resize_factor != 1:
                    crops = crops // args.dataset_resize_factor
                iterable = tqdm.tqdm(poses, desc="Pose", leave=False) if len(
                    poses) > 1 else poses
                for i, pose_num in enumerate(iterable):
                    t = ((self.train_data.num_frames - 1) *
                         pose_num / (args.dataset_render_poses - 1))
                    height = self.train_data.height
                    width = self.train_data.width
                    if self.train_data.using_processed_camera_parameters:
                        intrinsics = render_poses["intrinsics"][pose_num].to(
                            self.device)
                        transforms = render_poses["transforms"][pose_num].to(
                            self.device)
                        if args.inference_halves:
                            intrinsics = intrinsics.half()
                            transforms = transforms.half()
                        ray_origins, ray_directions = self.train_data.camera_params_to_rays(
                            intrinsics,
                            transforms,
                            height, width
                        )
                    else:
                        focal = self.train_data.focal
                        pose = render_poses[pose_num]
                        ray_origins, ray_directions = nerf_utils.get_ray_bundle(height, width, focal,
                                                                                pose[:3, :4].to(device))
                    if args.model == LEARNED_RAY:
                        rays_plucker = self.model.ray_to_indices(
                            ray_origins, ray_directions)
                    else:
                        rays_plucker = my_torch_utils.convert_rays_to_plucker(
                            ray_origins, ray_directions)
                    rays_with_t = torch.cat(
                        (rays_plucker, t *
                         torch.ones_like(rays_plucker[:, :, :1])),
                        dim=-1)
                    my_crop = crops[i]
                    rays_with_t = (
                        rays_with_t[my_crop[1]:my_crop[3], my_crop[0]:my_crop[2]]).half()
                    t0 = time.time()
                    rendered_image_outputs = self.render_full_image(batch_size,
                                                                    rays_with_t,
                                                                    lods=[lod],
                                                                    early_stopping=True,
                                                                    aux_discarding=True)
                    render_time = time.time() - t0
                    render_times.append(render_time)
                    image = rendered_image_outputs.full_image.float().cpu().numpy()
                    my_utils.join_and_make(
                        output_frames_dir, f"view{pose_num}")
                    output_filepath = os.path.join(output_frames_dir,
                                                   f"view{pose_num}", "%05d.png" % lod_index)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
                    num_rows_to_concatenate = int(
                        80 / args.dataset_resize_factor)
                    image = np.concatenate(
                        (image, np.zeros_like(image[:num_rows_to_concatenate])), axis=0)
                    height, width, _ = image.shape
                    font_scale = 2 * (1 / args.dataset_resize_factor)
                    color = (0, 0, 0, 1)
                    font_thickness = int(
                        max(1, 5 * (1 / args.dataset_resize_factor)))
                    text = f"LOD {lod+1:.2f}"
                    text_size, _ = cv2.getTextSize(
                        text, font, font_scale, font_thickness)
                    text_w, text_h = text_size
                    org = (int(0.5 * width - 0.5 * text_w),
                           int(height-0.5 * text_h))
                    image = cv2.putText(image, text, org, font,
                                        font_scale, color, font_thickness, cv2.LINE_AA)
                    cv2.imwrite(output_filepath, 255 * image)
                render_times_filepath = os.path.join(
                    args.checkpoints_dir, "slimnet_levels_render_times.txt")
            with open(render_times_filepath, "w") as f:
                f.write(str(render_times))
            plot_output_filepath = os.path.join(
                args.checkpoints_dir, "slimnet_levels_render_times.png")
            import matplotlib.pyplot as plt
            plt.plot(1000 * np.array(render_times))
            ax = plt.gca()
            ax.set_ylim([0.8 * 1000 * render_times[0],
                        1.2 * 1000 * render_times[-1]])
            plt.savefig(plot_output_filepath)

    def dump_model_sizes(self):
        model_sizes = []
        for i in range(self.model.num_outputs):
            model_sizes.append(self.model.num_params(i))
        factors = np.array(self.model.factors)
        model_widths = np.round(
            self.model.hidden_features * factors).astype(int).tolist()
        with open(os.path.join(self.args.checkpoints_dir, "model_sizes.json"), "w") as f:
            json.dump({
                "model_sizes": model_sizes,
                "model_widths": model_widths
            },                      f, indent=2)


if __name__ == "__main__":
    app = App()
    app.start()
