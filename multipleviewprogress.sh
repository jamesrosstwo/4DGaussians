workdir=$1
python scripts/extractimages.py multipleview/$workdir
colmap feature_extractor --database_path ./colmap_tmp/database.db --image_path ./colmap_tmp/images  --SiftExtraction.max_image_size 4096 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1
colmap exhaustive_matcher --database_path ./colmap_tmp/database.db
mkdir ./colmap_tmp/sparse
colmap mapper \
    --database_path ./colmap_tmp/database.db \
    --image_path ./colmap_tmp/images \
    --output_path ./colmap_tmp/sparse \
    --Mapper.ba_global_max_num_iterations 1000 \
    --Mapper.ba_global_function_tolerance 0.0001 \
    --log_to_stderr=1 \
mkdir ./data/multipleview/$workdir/sparse_
cp -r ./colmap_tmp/sparse/0/* ./data/multipleview/$workdir/sparse_

mkdir ./colmap_tmp/dense
colmap image_undistorter --image_path ./colmap_tmp/images --input_path ./colmap_tmp/sparse/0 --output_path ./colmap_tmp/dense --output_type COLMAP
colmap patch_match_stereo --workspace_path ./colmap_tmp/dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
colmap stereo_fusion --workspace_path ./colmap_tmp/dense --workspace_format COLMAP --input_type geometric --output_path ./colmap_tmp/dense/fused.ply