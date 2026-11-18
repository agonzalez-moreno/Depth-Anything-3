#!/usr/bin/env python
"""
Canopy Dataset for camera-conditioned depth estimation.

This dataset loads images, camera parameters, and optionally depth maps
from the Toulouse dataset structure. It uses pre-computed image pairs
from the image_pairs directory.

Only images with at least `num_pairs` valid pairs are used as master (query)
images. Images with fewer pairs are NOT dropped from the dataset: they can
still be used as pair (child) images of other masters.

Images listed in `excluded_stems` are completely ignored: they are used
neither as master (query) images nor as pair (child) images.

Dataset structure:
  /dataset_root/
    ├── input/
    │   └── <subset>/
    │       ├── images/              # RGB images: {stem}.tif
    │       ├── camera_extrinsics/  # Extrinsics: {stem}.yaml (3x4 or 4x4 matrices)
    │       ├── camera_intrinsics/  # Intrinsics: {stem}.yaml (3x3 matrices)
    │       └── image_pairs/         # Pre-computed image pairs: {stem}.yaml with scores
    └── reference/
        └── depth_images/
            └── <subset>/            # Ground truth depth maps (optional)

Usage:
    from canopy_dataset import CanopyDataset

    dataset = CanopyDataset(
        dataset_root='/path/to/dataset',
        subset='synthetic_test_672x448',
        img_size=672,
        num_pairs=5,
        use_depth=True,  # Set to False for inference
        # excluded_stems=None  -> disable the default exclusion list
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import yaml


# Images that must never be used, neither as master (query) nor as pair (child).
DEFAULT_EXCLUDED_STEMS = (
	"20FE31100x00010_01778_4_8",
	"20FE31100x00010_01778_4_7",
	"20FE31100x00008_03673_4_8",
	"20FE31100x00009_01781_5_6",
	"20FE31100x00008_03673_5_8",
	"20FE31100x00010_03690_4_7",
	"20FE31100x00010_01778_5_7",
	"20FE31100x00009_01781_5_5",
	"20FE31100x00008_01866_1_5",
	"20FE31100x00008_01866_1_6",
	"20FE31100x00008_01866_1_7",
	"20FE31100x00008_01866_1_8",
	"20FE31100x00008_01866_1_9",
	"20FE31100x00008_01866_2_6",
	"20FE31100x00008_01866_2_7",
	"20FE31100x00008_01866_2_8",
	"20FE31100x00008_01866_2_9",
	"20FE31100x00008_01866_3_6",
	"20FE31100x00008_01866_3_7",
	"20FE31100x00008_01866_3_8",
	"20FE31100x00008_01866_3_9",
	"20FE31100x00008_01866_4_6",
	"20FE31100x00008_01866_4_7",
	"20FE31100x00008_01866_4_8",
	"20FE31100x00008_01866_4_9"
)


class DataField:
    PARENT_NAME = "parent_name"
    IMAGES = "images"
    EXTRINSICS_C2W = "extrinsics_c2w"
    INTRINSICS = "intrinsics"
    DEPTHS = "depths"
    IS_SYNTHETIC = "is_synthetic"


def load_yaml(path):
    """Load a YAML file and return as Python object."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_camera_yaml(yaml_content):
    """Parse camera extrinsic YAML (3x4 matrix) to 4x4 camera-to-world matrix."""
    matrix_3x4 = []
    for row in yaml_content:
        matrix_3x4.append([float(x) for x in row])
    extrinsic = np.eye(4)
    extrinsic[:3, :] = np.array(matrix_3x4)
    return extrinsic


def parse_intrinsic_yaml(yaml_content):
    """Parse camera intrinsic YAML (3x3 matrix) to numpy array."""
    matrix = []
    for row in yaml_content:
        matrix.append([float(x) for x in row])
    return np.array(matrix)


class CanopyDataset(Dataset):
    """Dataset for camera-conditioned depth estimation.

    Loads images, camera parameters (extrinsics and intrinsics), and optionally
    depth maps.

    Two roles are distinguished:
    - Master (query) images: images with at least `num_pairs` valid pairs.
      Only these are indexed by `__len__` / `__getitem__`.
    - Pair (child) images: any image present on disk. An image that does not
      qualify as a master can still be used as a pair of another master.

    Images listed in `excluded_stems` have neither role.

    Each sample contains:
    - Query image (1 image)
    - Pair images (num_pairs images for multi-view context)
    - Camera extrinsics for all images (camera-to-world)
    - Camera intrinsics for all images
    - Depth map for all images (if use_depth=True)

    The dataset structure:
        dataset_root/
          ├── input/
          │   └── <subset>/
          │       ├── images/           # RGB images (*.tif)
          │       ├── camera_extrinsics/  # Camera-to-world extrinsics (*.yaml)
          │       ├── camera_intrinsics/   # Camera intrinsics (*.yaml)
          │       └── image_pairs/         # Pre-computed pairs with scores (*.yaml)
          └── reference/
              └── depth_images/
                  └── <subset>/         # Ground truth depth maps (*.tif)
    """

    def __init__(
        self,
        dataset_root,
        subset="synthetic_test_672x448",
        img_size=672,
        num_pairs=5,
        use_depth=True,
        excluded_stems=DEFAULT_EXCLUDED_STEMS,
    ):
        """Initialize the dataset.

        Args:
            dataset_root: Root path to the dataset
            subset: Subset name (e.g., 'synthetic_test_672x448')
            img_size: Target image size (no resizing, just for reference)
            num_pairs: Number of context images per query
            use_depth: Whether to load depth maps (for training)
            excluded_stems: Iterable of image names (stems, with or without the
                .tif extension) to ignore completely: they are used neither as
                master nor as pair images. Defaults to DEFAULT_EXCLUDED_STEMS.
                Pass None or an empty list to disable the exclusion.
        """
        self.dataset_root = Path(dataset_root)
        self.subset = subset
        self.img_size = img_size
        self.num_pairs = num_pairs
        self.use_depth = use_depth
        self.excluded_stems = {Path(str(s)).stem for s in (excluded_stems or [])}

        # Setup paths
        self.input_root = self.dataset_root / "input" / subset
        self.images_dir = self.input_root / "images"
        self.extrinsics_dir = self.input_root / "camera_extrinsics"
        self.intrinsics_dir = self.input_root / "camera_intrinsics"
        self.depth_root = self.dataset_root / "reference" / "depth_images" / subset
        self.image_pairs_dir = self.input_root / "image_pairs"

        found_image_files = sorted(self.images_dir.glob("*.tif"))
        if not found_image_files:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        # Drop excluded images: they can be neither master nor pair
        all_image_files = [f for f in found_image_files if f.stem not in self.excluded_stems]
        num_excluded = len(found_image_files) - len(all_image_files)
        if num_excluded:
            print(f"CanopyDataset: {num_excluded} image(s) excluded (not used as master nor pair)")
        missing_excluded = self.excluded_stems - {f.stem for f in found_image_files}
        if missing_excluded:
            print(f"Note: {len(missing_excluded)} excluded stem(s) not found in "
                  f"{self.images_dir}: {sorted(missing_excluded)}")
        if not all_image_files:
            raise FileNotFoundError(
                f"No images left in {self.images_dir} after exclusion"
            )

        # Every remaining image on disk can be used as a pair (child) image.
        # Excluded stems are absent from this set, so _read_pair_stems filters
        # them out of every master's pair list.
        self.all_stems = {f.stem for f in all_image_files}

        # Only images with >= num_pairs usable pairs become master images
        self.query_stems = []
        self._pairs_cache = {}  # master stem -> list of num_pairs pair stems, best score first

        for f in all_image_files:
            stem = f.stem
            pairs_path = self.image_pairs_dir / f"{stem}.yaml"
            if not pairs_path.exists():
                print(f"Warning: No image_pairs file for {stem}, not used as master")
                continue

            try:
                pair_stems = self._read_pair_stems(stem, pairs_path)
            except Exception as e:
                print(f"Warning: Failed to load pairs for {stem}: {e}, not used as master")
                continue

            if len(pair_stems) >= self.num_pairs:
                self.query_stems.append(stem)
                self._pairs_cache[stem] = pair_stems[:self.num_pairs]
            else:
                print(f"Warning: {stem} has only {len(pair_stems)} valid pairs "
                      f"(need {self.num_pairs}), not used as master")

        if not self.query_stems:
            raise FileNotFoundError(
                f"No images with at least {self.num_pairs} valid pairs found in {self.image_pairs_dir}"
            )

        print(f"CanopyDataset: {len(self.query_stems)} master images "
              f"(out of {len(self.all_stems)} images) with at least {self.num_pairs} pairs")

    def _read_pair_stems(self, stem, pairs_path):
        """Read the pairs of an image from its YAML file.

        Each YAML file contains a list of pairs with scores:
        - - score (float)
          - pair_image_name (string)

        Returns the pair stems sorted by score (descending), keeping only those
        that exist as images on disk (masters or not), are not excluded, are not
        the query itself, and are unique.
        """
        entries = load_yaml(pairs_path)
        scored = [(float(e[0]), str(e[1])) for e in entries
                  if isinstance(e, list) and len(e) >= 2]
        scored.sort(key=lambda x: x[0], reverse=True)

        seen, result = set(), []
        for _, pair_stem in scored:
            if pair_stem == stem or pair_stem in seen:
                continue
            # all_stems excludes the images listed in excluded_stems
            if pair_stem not in self.all_stems:
                continue
            seen.add(pair_stem)
            result.append(pair_stem)
        return result

    def __len__(self):
        """Return the number of master (query) images in the dataset."""
        return len(self.query_stems)

    def _load_image(self, stem):
        """Load an image from disk."""
        img_path = self.images_dir / f"{stem}.tif"
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def _load_depth(self, stem):
        """Load a depth map from disk. Returns None if not found."""
        depth_path = self.depth_root / f"{stem}.tif"
        if not depth_path.exists():
            return None
        depth_img = Image.open(depth_path)
        depth_array = np.array(depth_img, dtype=np.float32)
        depth = depth_array / 1400.0
        return depth

    def _load_extrinsic(self, stem):
        """Load camera extrinsic matrix (camera-to-world) from YAML."""
        ext_path = self.extrinsics_dir / f"{stem}.yaml"
        yaml_content = load_yaml(ext_path)
        extrinsics = parse_camera_yaml(yaml_content)
        extrinsics[:3,3] = extrinsics[:3,3] / 1400.0
        return parse_camera_yaml(yaml_content)

    def _load_intrinsic(self, stem):
        """Load camera intrinsic matrix from YAML."""
        int_path = self.intrinsics_dir / f"{stem}.yaml"
        yaml_content = load_yaml(int_path)
        return parse_intrinsic_yaml(yaml_content)

    def _preprocess_image(self, img):
        """Convert PIL Image to tensor and normalize.

        Args:
            img: PIL Image (RGB)

        Returns:
            Tensor of shape [C, H, W] with values in [0, 1]
        """
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    def _preprocess_depth(self, depth):
        """Preprocess depth map. Returns as-is (no resizing).

        Args:
            depth: Depth map as numpy array, or None

        Returns:
            Depth map as numpy array, or None
        """
        return depth

    def __getitem__(self, idx):
        """Get a batch of data for a single master (query) image.

        Returns:
            dict with keys:
                - DataField.PARENT_NAME: string
                - DataField.IMAGES: [S, C, H, W] tensor (S = 1 query + num_pairs)
                - DataField.EXTRINSICS_C2W: [S, 4, 4] tensor
                - DataField.INTRINSICS: [S, 3, 3] tensor
                - DataField.DEPTHS: [S, H, W] tensor (if use_depth=True)
                - DataField.IS_SYNTHETIC: [1] tensor (boolean)
        """
        query_stem = self.query_stems[idx]
        all_stems = [query_stem] + self._pairs_cache[query_stem]

        # Load and convert images
        images = [self._preprocess_image(self._load_image(stem)) for stem in all_stems]
        images_tensor = torch.stack(images, dim=0)  # [S, C, H, W]

        # Load camera extrinsics (camera-to-world)
        mean_plane_altitude = 1400.0
        extrinsics_c2w = [self._load_extrinsic(stem)[:4, :4] for stem in all_stems]
        extrinsics_tensor = torch.from_numpy(np.stack(extrinsics_c2w)).float()  # [S, 4, 4]

        # Load camera intrinsics
        intrinsics = [self._load_intrinsic(stem) for stem in all_stems]
        intrinsics_tensor = torch.from_numpy(np.stack(intrinsics)).float()  # [S, 3, 3]

        batch = {
            DataField.PARENT_NAME: query_stem,
            DataField.IMAGES: images_tensor,
            DataField.EXTRINSICS_C2W: extrinsics_tensor,
            DataField.INTRINSICS: intrinsics_tensor,
            DataField.IS_SYNTHETIC: torch.ones(1, dtype=torch.bool),
        }

        # Load depth maps if requested
        if self.use_depth:
            depths = []
            for stem in all_stems:
                depth = self._preprocess_depth(self._load_depth(stem))
                if depth is None:
                    raise FileNotFoundError(
                        f"Depth map not found for {stem} in {self.depth_root}"
                    )
                depths.append(depth)
            batch[DataField.DEPTHS] = torch.from_numpy(np.stack(depths)).float()  # [S, H, W]

        return batch
