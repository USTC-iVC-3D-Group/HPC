#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np

import torch
import torchvision
import json
import time
import logging

from os import makedirs

from random import randint
from utils.metrics import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render
import sys
from scene import Scene, P_Scene, GaussianModel
from utils.common import safe_state
from tqdm import tqdm
from utils.metrics import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.encodings import get_binary_vxl_size

from lpipsPyTorch import lpips

bit2MB_scale = 8 * 1024 * 1024

import random

torch.backends.cudnn.benchmark = True

def training_I_frame(args_param, dataset, opt, pipe, testing_iterations, saving_iterations, logger=None, ply_path=None):
    first_iter = 0
    prepare_output_and_logger(dataset)

    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        n_features_per_level=args_param.n_features,
        log2_hashmap_size=args_param.log2,
        log2_hashmap_size_2D=args_param.log2_2D,
        mode = "I_frame",
        enable_filter = dataset.enable_filter,
        frame_idx = dataset.frame_idx,
    )
    scene = Scene(dataset, gaussians, ply_path=ply_path)

    gaussians.update_anchor_bound()
    gaussians.set_steps(args_param.step_flag1, args_param.step_flag2)
    gaussians.training_setup(opt)

    logger.info(f"train enable_filter: {gaussians.enable_filter}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    torch.cuda.synchronize(); t_start = time.time()
    log_time_sub = 0
    for iteration in range(first_iter, opt.iterations + 1):

        gaussians.update_learning_rate(iteration)

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)

        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, step=iteration)
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        bit_per_param = render_pkg["bit_per_param"]
        bit_per_feat_param = render_pkg["bit_per_feat_param"]
        bit_per_scaling_param = render_pkg["bit_per_scaling_param"]
        bit_per_offsets_param = render_pkg["bit_per_offsets_param"]

        if iteration % 2000 == 0 and bit_per_param is not None:

            ttl_size_feat_MB = bit_per_feat_param.item() * gaussians.get_anchor.shape[0] * gaussians.feat_dim / bit2MB_scale
            ttl_size_scaling_MB = bit_per_scaling_param.item() * gaussians.get_anchor.shape[0] * 6 / bit2MB_scale
            ttl_size_offsets_MB = bit_per_offsets_param.item() * gaussians.get_anchor.shape[0] * 3 * gaussians.n_offsets / bit2MB_scale
            ttl_size_MB = ttl_size_feat_MB + ttl_size_scaling_MB + ttl_size_offsets_MB

            logger.info("\n----------------------------------------------------------------------------------------")
            logger.info("\n-----[ITER {}] bits info: bit_per_feat_param={}, anchor_num={}, ttl_size_feat_MB={}-----".format(iteration, bit_per_feat_param.item(), gaussians.get_anchor.shape[0], ttl_size_feat_MB))
            logger.info("\n-----[ITER {}] bits info: bit_per_scaling_param={}, anchor_num={}, ttl_size_scaling_MB={}-----".format(iteration, bit_per_scaling_param.item(), gaussians.get_anchor.shape[0], ttl_size_scaling_MB))
            logger.info("\n-----[ITER {}] bits info: bit_per_offsets_param={}, anchor_num={}, ttl_size_offsets_MB={}-----".format(iteration, bit_per_offsets_param.item(), gaussians.get_anchor.shape[0], ttl_size_offsets_MB))
            logger.info("\n-----[ITER {}] bits info: bit_per_param={}, anchor_num={}, ttl_size_MB={}-----".format(iteration, bit_per_param.item(), gaussians.get_anchor.shape[0], ttl_size_MB))

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

        if bit_per_param is not None:
            _, bit_hash_grid, MB_hash_grid, _ = get_binary_vxl_size((gaussians.get_encoding_params()+1)/2)
            denom = gaussians._anchor.shape[0]*(gaussians.feat_dim+6+3*gaussians.n_offsets)
            loss = loss + args_param.lmbda * (bit_per_param + bit_hash_grid / denom)

        loss.backward()

        with torch.no_grad():

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            torch.cuda.synchronize(); t_start_log = time.time()
            training_report(iteration, l1_loss, testing_iterations, scene, render, (pipe, background), logger, args_param.model_path)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            torch.cuda.synchronize(); t_end_log = time.time()
            t_log = t_end_log - t_start_log
            log_time_sub += t_log

            if iteration < opt.update_until and iteration > opt.start_stat:
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                if iteration not in range(3000, 4000):  # let the model get fit to quantization

                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

    torch.cuda.synchronize(); t_end = time.time()
    logger.info("\n Total Training time: {}".format(t_end-t_start-log_time_sub))

def training_P_frame(args_param, dataset, opt, pipe, testing_iterations, saving_iterations, logger=None, ply_path=None):
    first_iter = 0
    prepare_output_and_logger(dataset)

    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        n_features_per_level=args_param.n_features,
        log2_hashmap_size=args_param.log2,
        log2_hashmap_size_2D=args_param.log2_2D,
        decoded_version=True, # True
        mode="P_frame",
        enable_filter = dataset.enable_filter,
        ref_path = dataset.ref_path,
        ref_iter = dataset.ref_iter,
        frame_idx = dataset.frame_idx,
    )
    scene = P_Scene(dataset, gaussians, ply_path=ply_path)
    gaussians.gop_size = args_param.gop_size

    gaussians.training_setup_for_HPC(args_param.hpc_cfg, args_param.nn_bitdepth)
    gaussians.train()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    torch.cuda.synchronize(); t_start = time.time()
    log_time_sub = 0
    for iteration in range(first_iter, opt.iterations + 1):

        gaussians.update_learning_rate_for_P_frame(iteration)

        nn_bpp = 0.0
        nn_bpp = gaussians.nn_bits_estimation()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, step=iteration)
        image, viewspace_point_tensor, visibility_filter, radii, bit_per_latent = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["bit_per_latent"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss

        if bit_per_latent is not None:
            loss = loss + args_param.lmbda * args_param.P_lmbda * (bit_per_latent + nn_bpp)

        loss.backward()

        with torch.no_grad():

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            torch.cuda.synchronize(); t_start_log = time.time()
            training_report(iteration, l1_loss, testing_iterations, scene, render, (pipe, background), logger, args_param.model_path)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            torch.cuda.synchronize(); t_end_log = time.time()
            t_log = t_end_log - t_start_log
            log_time_sub += t_log

            if iteration < opt.iterations:
                gaussians.ntc_optimizer.step()
                gaussians.ntc_optimizer.zero_grad(set_to_none = True)

    torch.cuda.synchronize(); t_end = time.time()
    logger.info("\n Total Training time: {}".format(t_end-t_start-log_time_sub))

def prepare_output_and_logger(args):

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def training_report(iteration, l1_loss, testing_iterations, scene, renderFunc, renderArgs, logger=None, pre_path_name=''):

    if iteration in testing_iterations:
        scene.gaussians.eval()
        mode = scene.gaussians.mode

        if iteration == testing_iterations[-1]:
            if mode == "I_frame":
                with torch.no_grad():
                    log_info = scene.gaussians.estimate_final_bits()
                    logger.info(log_info)
            with torch.no_grad():
                bit_stream_path = os.path.join(pre_path_name, f"iteration_{iteration}",'bitstreams')
                os.makedirs(bit_stream_path, exist_ok=True)
                if mode == "I_frame":

                    log_info = scene.gaussians.conduct_encoding(pre_path_name=bit_stream_path)
                    logger.info(log_info)

                    log_info = scene.gaussians.conduct_decoding(pre_path_name=bit_stream_path)
                    logger.info(log_info)
                elif mode == "P_frame":

                    log_info = scene.gaussians.conduct_encoding_for_hpc(pre_path_name=bit_stream_path)
                    logger.info(log_info)

                    log_info = scene.gaussians.conduct_decoding_for_hpc(pre_path_name=bit_stream_path)
                    logger.info(log_info)
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:

            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0

                t_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    torch.cuda.synchronize(); t_start = time.time()
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)

                    render_output = renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)
                    time_sub = render_output["time_sub"]
                    torch.cuda.synchronize();
                    t_end = time.time()
                    t_list.append(t_end - t_start - time_sub)
                    image = torch.clamp(render_output["render"], 0.0, 1.0)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    lpips_test += lpips(image, gt_image, net_type='vgg').detach().mean().double()

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {} ssim {} lpips {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                test_fps = 1.0 / torch.tensor(t_list[0:]).mean()
                logger.info(f'Test FPS: {test_fps.item():.5f}')

        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    t_list = []
    per_view_dict = {}
    psnr_list = []
    ssim_list = []
    lpips_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t_start = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t_end = time.time()
        time_sub = render_pkg['time_sub']

        t_list.append(t_end - t_start - time_sub)

        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()

        gt = view.original_image[0:3, :, :]

        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        render_image = torch.clamp(rendering.to("cuda"), 0.0, 1.0)
        psnr_view = psnr(render_image, gt_image).mean().double()
        psnr_list.append(psnr_view)
        ssim_view = ssim(render_image, gt_image).mean().double()
        ssim_list.append(ssim_view)
        lpips_view = lpips(render_image, gt_image, net_type='vgg').detach().mean().double()
        lpips_list.append(lpips_view)

        errormap = (rendering - gt).abs()

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

    metric = [psnr_list, ssim_list, lpips_list]

    return t_list, metric

def render_sets(args_param, dataset : ModelParams, iteration : int, pipeline : PipelineParams, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            n_features_per_level=args_param.n_features,
            log2_hashmap_size=args_param.log2,
            log2_hashmap_size_2D=args_param.log2_2D,
            decoded_version=True,
            enable_filter = dataset.enable_filter,
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        logger.info(f"test enable_filter: {gaussians.enable_filter}")

        t_test_list, test_metric = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

        test_fps = 1.0 / torch.tensor(t_test_list).mean()
        logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')

    return test_metric

def evaluate(model_paths, test_metric, iterations, logger=None):

    full_dict = {}
    print("")

    scene_dir = model_paths
    full_dict[scene_dir] = {}

    method = "ours_{}".format(iterations)

    full_dict[scene_dir][method] = {}

    psnrs, ssims, lpipss = test_metric

    logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
    logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
    logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
    logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
    print("")

    full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                            "LPIPS": torch.tensor(lpipss).mean().item()})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)

    psnr_f = torch.tensor(psnrs).mean().item()
    ssim_f = torch.tensor(ssims).mean().item()
    lpips_f = torch.tensor(lpipss).mean().item()

    return psnr_f, ssim_f, lpips_f

def get_logger(path):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

def release_logger():

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

if __name__ == "__main__":

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log2", type=int, default = 13)
    parser.add_argument("--log2_2D", type=int, default = 15)
    parser.add_argument("--n_features", type=int, default = 4)
    parser.add_argument("--config_path", type=str, default = None)
    parser.add_argument("--experiment", type=str, default = None)
    parser.add_argument("--nn_bitdepth", type=int, default=8, choices=range(1, 9))
    parser.add_argument("--I_lmbda", type=float, default = None)
    parser.add_argument("--P_lmbda", type=float, default = None)
    parser.add_argument('--data-root', type=str)
    parser.add_argument('--output-root', type=str)
    parser.add_argument('--frame-start', type=int)
    parser.add_argument('--frame-end', type=int)
    parser.add_argument('--gop-size', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args(sys.argv[1:])

    safe_state(args.quiet)

    assert args.config_path is not None, "Please provide a config path"
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    if args.gop_size < 1:
        parser.error('--gop-size must be positive')
    if args.data_root is not None:
        config['base_source_path'] = args.data_root
    if args.output_root is not None:
        config['base_model_path'] = args.output_root
    if args.frame_start is not None:
        config['frame_start'] = args.frame_start
    if args.frame_end is not None:
        config['frame_end'] = args.frame_end
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    scene = config["scene"]
    frame_start = config["frame_start"]
    frame_end = config["frame_end"]
    lmbda = args.I_lmbda if args.I_lmbda is not None else config['lmbda']
    P_lmbda = args.P_lmbda if args.P_lmbda is not None else config['P_lmbda']

    base_source_path = config["base_source_path"]
    base_model_path = config["base_model_path"]
    if args.experiment is not None:
        base_model_path = os.path.join(base_model_path, args.experiment)

    psnr_list, ssim_list, lpips_list = {}, {}, {}
    psnr_sum, ssim_sum, lpips_sum = 0, 0, 0

    for frame in range(frame_start, frame_end+1):

        np.random.randint(10000, 20000)  # Keep the established NumPy initialization sequence.
        torch.autograd.set_detect_anomaly(args.detect_anomaly)

        print("Frame:", frame)

        for key, value in config["model_params"].items():
                setattr(args, key, value)
        args.source_path = os.path.join(base_source_path, scene, f"frame{frame:06d}")
        dataset = args.source_path.split('/')[-1]
        args.lmbda = lmbda
        args.frame_idx = frame

        if frame == 0:
            args.model_path = os.path.join(base_model_path, scene, f"{lmbda}_{P_lmbda}", f"frame{frame:06d}")
            os.makedirs(args.model_path, exist_ok=True)
            logger = get_logger(args.model_path)

            for key, value in config["Init_frame_params"].items():
                setattr(args, key, value)

            logger.info("Optimizing " + args.model_path)
            logger.info(f'args: {args}')

            print(args.test_iterations)
            training_I_frame(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, logger)

        else:
            config_P = config["P_frame_params"]

            args.hpc_cfg = config_P["hpc_cfg"]
            args.P_lmbda = P_lmbda

            args.model_path = os.path.join(base_model_path, scene, f"{lmbda}_{P_lmbda}", f"frame{frame:06d}")
            os.makedirs(args.model_path, exist_ok=True)
            logger = get_logger(args.model_path)

            for key, value in config_P.items():
                if key != "first_ref_iteration" and key != "hpc_cfg":
                    setattr(args, key, value)
            args.ref_path = os.path.join(base_model_path, scene, f"{lmbda}_{P_lmbda}", f"frame{frame-1:06d}")
            args.ref_iter = (config_P['first_ref_iteration'] if frame == 1
                             else config_P['ref_iter'])

            logger.info("Optimizing " + args.model_path)
            logger.info(f'args: {args}')
            training_P_frame(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, logger)

        logger.info("\nTraining complete.")

        logger.info(f'\nStarting Rendering and evaluation~')
        for iteration in args.save_iterations:
            test_metric = render_sets(args, lp.extract(args), iteration, pp.extract(args), logger=logger)
        logger.info("\nRendering and evaluation complete.")

        logger.info("\n Starting evaluation logging...")
        psnr_f, ssim_f, lpips_f = evaluate(args.model_path, test_metric, iteration, logger=logger)
        psnr_list[f"{frame:06d}"] = psnr_f
        ssim_list[f"{frame:06d}"] = ssim_f
        lpips_list[f"{frame:06d}"] = lpips_f
        psnr_sum += psnr_f
        ssim_sum += ssim_f
        lpips_sum += lpips_f
        logger.info("\nEvaluating complete.")

        release_logger()

    num_frames = frame_end - frame_start + 1
    mean_list = {"psnr":(psnr_sum/num_frames), "ssim":(ssim_sum/num_frames), "lpips":(lpips_sum/num_frames)}
    with open(os.path.join(base_model_path, scene, f"{lmbda}_{P_lmbda}", f"frames_results.json"), 'w') as fp:
        json.dump({"mean":mean_list, "psnr":psnr_list, "ssim":ssim_list, "lpips":lpips_list}, fp, indent=True)
