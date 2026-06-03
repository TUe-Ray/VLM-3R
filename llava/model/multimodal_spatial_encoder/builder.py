import os
import sys

# Get the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up four levels to reach VLM-3R/
vlm_3r_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))


def _first_existing_dir(*relative_paths):
    for rel_path in relative_paths:
        abs_path = os.path.join(vlm_3r_root, rel_path)
        if os.path.isdir(abs_path):
            return abs_path
    return os.path.join(vlm_3r_root, relative_paths[0])

def build_spatial_tower(spatial_tower_cfg, **kwargs):
    spatial_tower = getattr(spatial_tower_cfg, "mm_spatial_tower", getattr(spatial_tower_cfg, "spatial_tower", 'spann3r'))
    if spatial_tower == "spann3r":
        spann3r_path = os.path.join(vlm_3r_root, 'spann3r')
        if spann3r_path not in sys.path:
             sys.path.append(spann3r_path)
        # Use relative import for the encoder wrapper/adapter file
        from .spann3r_spatial_encoder import Spann3rSpatialTower
        return Spann3rSpatialTower(spatial_tower, spatial_tower_cfg=spatial_tower_cfg, **kwargs)
    elif spatial_tower == "cut3r":
        cut3r_path = _first_existing_dir('third_party/CUT3R', 'CUT3R')
        if cut3r_path not in sys.path:
            sys.path.append(cut3r_path)
        # Use relative import for the encoder wrapper/adapter file
        from .cut3r_spatial_encoder import Cut3rSpatialTower
        return Cut3rSpatialTower(spatial_tower, spatial_tower_cfg=spatial_tower_cfg, **kwargs)
    elif spatial_tower == "vggt":
        vggt_path = _first_existing_dir('third_party/VGGT', 'vggt', 'VGGT')
        if vggt_path not in sys.path:
            sys.path.append(vggt_path)
        # Use relative import for the encoder wrapper/adapter file
        from .vggt_spatial_encoder import VGGTSpatialTower
        return VGGTSpatialTower(spatial_tower, spatial_tower_cfg=spatial_tower_cfg, **kwargs)
    elif spatial_tower == "pi3x":
        pi3x_path = _first_existing_dir('third_party/Pi3', 'pi3x', 'Pi3')
        if pi3x_path not in sys.path:
            sys.path.append(pi3x_path)
        from .pi3x_spatial_encoder import Pi3xSpatialTower
        return Pi3xSpatialTower(spatial_tower, spatial_tower_cfg=spatial_tower_cfg, **kwargs)
    elif spatial_tower == "cut3r_points":
        cut3r_path = _first_existing_dir('third_party/CUT3R', 'CUT3R')
        if cut3r_path not in sys.path:
            sys.path.append(cut3r_path)
        # Use relative import for the encoder wrapper/adapter file
        from .cut3r_points import Cut3rPointsSpatialTower
        return Cut3rPointsSpatialTower(spatial_tower, spatial_tower_cfg=spatial_tower_cfg, **kwargs)
    raise ValueError(f"Unknown vision tower: {spatial_tower}")
