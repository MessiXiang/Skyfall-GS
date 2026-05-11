# python train.py \
#     -s ./data/datasets_JAX/JAX_068/ \
#     -m ./outputs/JAX_idu/JAX_068_ablation_adaptive_position_only \
#     --start_checkpoint ./outputs/JAX/JAX_068/chkpnt30000.pth \
#     --iterative_datasets_update \
#     --eval \
#     --port 6209 \
#     --kernel_size 0.1 \
#     --resolution 1 \
#     --sh_degree 1 \
#     --appearance_enabled \
#     --lambda_depth 0 \
#     --lambda_opacity 0 \
#     --idu_opacity_reset_interval 5000 \
#     --idu_refine \
#     --idu_num_samples_per_view 1 \
#     --densify_grad_threshold 0.0002 \
#     --idu_use_flow_edit \
#     --idu_render_size 1024 \
#     --idu_flow_edit_n_min 4 \
#     --idu_flow_edit_n_max 14 \
#     --idu_grid_width 512 \
#     --idu_grid_height 512 \
#     --idu_adaptive_segformer_sampling \
#     --idu_segformer_model_name wu-pr-gw/segformer-b2-finetuned-with-LoveDA \
#     --idu_adaptive_seg_render_size 1024 \
#     --idu_adaptive_overview_fov 70 \
#     --idu_adaptive_overview_radius 1.0 \
#     --idu_adaptive_overview_radius_scale 0.65 \
#     --idu_adaptive_building_subdivisions 2 \
#     --idu_adaptive_other_subdivisions 1 \
#     --idu_adaptive_building_radius_scale 0.85 \
#     --idu_adaptive_other_radius_scale 1.0 \
#     --idu_adaptive_max_targets 64 \
#     --idu_adaptive_fine_grid_size 256 \
#     --idu_adaptive_coverage_cells 20 \
#     --idu_adaptive_building_weight 1.0 \
#     --idu_adaptive_road_weight 0.3 \
#     --idu_adaptive_wild_weight 0.1 \
#     --idu_adaptive_nms_radius_cells 10 \
#     --idu_episode_iterations 10000 \
#     --idu_iter_full_train 0 \
#     --idu_opacity_cooling_iterations 500 \
#     --lambda_pseudo_depth 0.5 \
#     --idu_densify_until_iter 9000 \
#     --idu_train_ratio 0.75 \
#     --idu_depth_estimator moge \
#     --idu_use_sr \
#     --idu_sr_method lanczos \
#     --idu_sr_scale 2 \
#     --idu_sr_downsample_back \
#     --idu_sr_post_sharpen_percent 80 \
#     --idu_sr_post_sharpen_radius 0.4 \
#     --idu_sr_post_sharpen_threshold 2 \
#     --idu_adaptive_building_four_direction_views

# python train.py \
#     -s ./data/datasets_JAX/JAX_068/ \
#     -m ./outputs/JAX_idu/JAX_068_ablation_adaptive_angle_only \
#     --start_checkpoint ./outputs/JAX/JAX_068/chkpnt30000.pth \
#     --iterative_datasets_update \
#     --eval \
#     --port 6209 \
#     --kernel_size 0.1 \
#     --resolution 1 \
#     --sh_degree 1 \
#     --appearance_enabled \
#     --lambda_depth 0 \
#     --lambda_opacity 0 \
#     --idu_opacity_reset_interval 5000 \
#     --idu_refine \
#     --idu_num_samples_per_view 1 \
#     --densify_grad_threshold 0.0002 \
#     --idu_use_flow_edit \
#     --idu_render_size 1024 \
#     --idu_flow_edit_n_min 4 \
#     --idu_flow_edit_n_max 14 \
#     --idu_grid_width 512 \
#     --idu_grid_height 512 \
#     --idu_episode_iterations 10000 \
#     --idu_iter_full_train 0 \
#     --idu_opacity_cooling_iterations 500 \
#     --lambda_pseudo_depth 0.5 \
#     --idu_densify_until_iter 9000 \
#     --idu_train_ratio 0.75 \
#     --idu_depth_estimator moge \
#     --idu_use_sr \
#     --idu_sr_method lanczos \
#     --idu_sr_scale 2 \
#     --idu_sr_downsample_back \
#     --idu_sr_post_sharpen_percent 80 \
#     --idu_sr_post_sharpen_radius 0.4 \
#     --idu_sr_post_sharpen_threshold 2 \
#     --idu_knee_elevation_sampling \
#     --idu_knee_use_global_range \
#     --idu_knee_min_elevation 20 \
#     --idu_knee_max_elevation 85 \
#     --idu_knee_candidate_step 5 \
#     --idu_knee_quality_alpha 3.0 \
#     --idu_knee_info_beta 0.8 \
#     --idu_knee_render_size 256 \
#     --idu_knee_select_mode balance \
#     --idu_knee_metric_mode coverage \
#     --idu_knee_missing_penalty 1.5

python train.py \
    -s ./data/datasets_JAX/JAX_068/ \
    -m ./outputs/JAX_idu/JAX_068_ablation_no_adaptive_position_no_adaptive_angle \
    --start_checkpoint ./outputs/JAX/JAX_068/chkpnt30000.pth \
    --iterative_datasets_update \
    --eval \
    --port 6209 \
    --kernel_size 0.1 \
    --resolution 1 \
    --sh_degree 1 \
    --appearance_enabled \
    --lambda_depth 0 \
    --lambda_opacity 0 \
    --idu_opacity_reset_interval 5000 \
    --idu_refine \
    --idu_num_samples_per_view 1 \
    --densify_grad_threshold 0.0002 \
    --idu_use_flow_edit \
    --idu_render_size 1024 \
    --idu_flow_edit_n_min 4 \
    --idu_flow_edit_n_max 14 \
    --idu_grid_width 512 \
    --idu_grid_height 512 \
    --idu_episode_iterations 10000 \
    --idu_iter_full_train 0 \
    --idu_opacity_cooling_iterations 500 \
    --lambda_pseudo_depth 0.5 \
    --idu_densify_until_iter 9000 \
    --idu_train_ratio 0.75 \
    --idu_depth_estimator moge \
    --idu_use_sr \
    --idu_sr_method lanczos \
    --idu_sr_scale 2 \
    --idu_sr_downsample_back \
    --idu_sr_post_sharpen_percent 20 \
    --idu_sr_post_sharpen_radius 0.1 \
    --idu_sr_post_sharpen_threshold 2

echo "Ablation studies completed."