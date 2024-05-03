from pathlib import Path
import pycolmap
import pandas as pd
import torch
from scripts.extract_keypoint import detect_keypoints
from scripts.image_pair import get_image_pairs
from scripts.keypoint_distance import keypoint_distances
from scripts.match import visualize_matches
from scripts.ransac import import_into_colmap
from scripts.utils import plot_reconstruction,colmap_dense_reconstruction,save_rot_tra_info
import os
base_dir=os.getcwd()
scene='bike'
dataset='haiper'
PATH = os.path.join(base_dir,'train',f'{dataset}',f'{scene}','images')
# PATH=f'/home/ubuntu/Final-Project-Group2/Code/train/{dataset}/{scene}/images'
print(PATH)
EXT = 'jpeg'
PATH_FEATURES = os.path.join(base_dir,'features')
DINO_PATH = os.path.join(base_dir,'dinov2','pytorch','base','1')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get Image Pairs for Matching using DINO
print(Path(PATH).glob(f'*.{EXT}'))
images_list = list(Path(PATH).glob(f'*.{EXT}'))
index_pairs = get_image_pairs(images_list, DINO_PATH)

# Extract keypoints using ALIKED
feature_dir = Path(PATH_FEATURES)
detect_keypoints(images_list, feature_dir, device=device)

# Compute Keypoint Distances  using LightGlue
keypoint_distances(images_list, index_pairs, feature_dir, verbose=False, device=device)

# Visualise Image matching
idx1, idx2 = index_pairs[0]
visualize_matches(images_list, idx1, idx2, feature_dir)

base_dir = Path.cwd()

database_path = base_dir / f'colmap_{scene}.db'
# Ensure database path does not exist before continuing
if database_path.exists():
    database_path.unlink()

images_dir = images_list[0].parent
import_into_colmap(
    images_dir,
    feature_dir,
    database_path,
)

# This does RANSAC
pycolmap.match_exhaustive(database_path)

# This does the reconstruction
mapper_options = pycolmap.IncrementalPipelineOptions()
mapper_options.min_model_size = 3
mapper_options.max_num_models = 2

maps = pycolmap.incremental_mapping(
    database_path=database_path,
    image_path=images_dir,
    output_path=Path.cwd() / "incremental_pipeline_outputs",
    options=mapper_options,
)

# Save rotational matrix and translation info into a csv file
save_rot_tra_info(maps, 'rot_tra_info.csv')

# Visualize the 3D reconstruction
plot_reconstruction(maps[0], f'Reconstruction_{scene}.html')

#
# # Dense Reconstruction
# # colmap_dense_reconstruction(images_dir, database_path, Path.cwd())

