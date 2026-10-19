"""ImageNet-100 dataset support.

ImageNet-100 does not have one universally agreed class subset.  This loader
therefore treats the directory tree as the definition of the subset and fails
fast unless train and validation contain the same 100 class directories.
"""

import hashlib
from pathlib import Path

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNet100:
    """Load a materialized 100-class ImageNet subset in ImageFolder format."""

    def __init__(
        self,
        root="./data/imagenet100",
        train_subdir="train",
        val_subdir="val",
        image_size=224,
        crop_pct=0.875,
        recipe="full",
        randaugment_num_ops=2,
        randaugment_magnitude=9,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        train_crop_scale=(0.08, 1.0),
        train_crop_ratio=(0.75, 1.3333333333333333),
        horizontal_flip_p=0.5,
        random_erasing_p=0.25,
        random_erasing_mode="pixel",
        random_erasing_count=1,
        class_manifest_sha256=None,
    ):
        root = Path(root).expanduser()
        train_dir = root / train_subdir
        val_dir = root / val_subdir
        self._validate_split_directory(train_dir, "training")
        self._validate_split_directory(val_dir, "validation")

        image_size = int(image_size)
        crop_pct = float(crop_pct)
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if not 0.0 < crop_pct <= 1.0:
            raise ValueError(f"crop_pct must be in (0, 1], got {crop_pct}")
        if recipe not in {"base", "full", "official"}:
            raise ValueError(
                "ImageNet100 recipe must be 'base', 'full', or 'official'"
            )

        if recipe == "official":
            try:
                from timm.data import create_transform
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "The official TinyViT augmentation recipe requires timm. "
                    "Install requirements-imagenet100.txt."
                ) from exc

            train_transform = create_transform(
                input_size=image_size,
                is_training=True,
                color_jitter=float(color_jitter),
                auto_augment=str(auto_augment),
                interpolation=str(interpolation),
                scale=tuple(float(v) for v in train_crop_scale),
                ratio=tuple(float(v) for v in train_crop_ratio),
                hflip=float(horizontal_flip_p),
                vflip=0.0,
                re_prob=float(random_erasing_p),
                re_mode=str(random_erasing_mode),
                re_count=int(random_erasing_count),
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            )
        else:
            train_transform = self._build_torchvision_train_transform(
                image_size=image_size,
                recipe=recipe,
                randaugment_num_ops=randaugment_num_ops,
                randaugment_magnitude=randaugment_magnitude,
                random_erasing_p=random_erasing_p,
            )

        resize_size = int(round(image_size / crop_pct))
        eval_transform = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        self.trainset = ImageFolder(train_dir, transform=train_transform)
        self.trainset_eval = ImageFolder(train_dir, transform=eval_transform)
        self.testset = ImageFolder(val_dir, transform=eval_transform)
        self._validate_classes()
        manifest = "\n".join(self.trainset.classes).encode("utf-8")
        self.class_manifest_sha256 = hashlib.sha256(manifest).hexdigest()
        if (
            class_manifest_sha256 is not None
            and self.class_manifest_sha256 != str(class_manifest_sha256).lower()
        ):
            raise ValueError(
                "ImageNet-100 class manifest fingerprint mismatch: expected "
                f"{class_manifest_sha256}, found {self.class_manifest_sha256}"
            )

    @staticmethod
    def _build_torchvision_train_transform(
        image_size,
        recipe,
        randaugment_num_ops,
        randaugment_magnitude,
        random_erasing_p,
    ):
        train_ops = [
            transforms.RandomResizedCrop(
                image_size,
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
        ]
        if recipe == "full":
            train_ops.append(
                transforms.RandAugment(
                    num_ops=int(randaugment_num_ops),
                    magnitude=int(randaugment_magnitude),
                    interpolation=InterpolationMode.BICUBIC,
                )
            )
        train_ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        if recipe == "full" and float(random_erasing_p) > 0.0:
            train_ops.append(
                transforms.RandomErasing(
                    p=float(random_erasing_p),
                    value="random",
                )
            )

        return transforms.Compose(train_ops)

    @staticmethod
    def _validate_split_directory(path: Path, split_name: str) -> None:
        if not path.is_dir():
            raise FileNotFoundError(
                f"ImageNet-100 {split_name} directory not found: {path}. "
                "Expected <root>/train/<class> and <root>/val/<class>."
            )

    def _validate_classes(self) -> None:
        train_classes = self.trainset.classes
        val_classes = self.testset.classes
        if len(train_classes) != 100:
            raise ValueError(
                f"ImageNet-100 training split must contain exactly 100 class "
                f"directories, found {len(train_classes)}"
            )
        if train_classes != val_classes:
            missing_from_val = sorted(set(train_classes) - set(val_classes))
            extra_in_val = sorted(set(val_classes) - set(train_classes))
            raise ValueError(
                "ImageNet-100 train/validation class directories do not match. "
                f"Missing from val: {missing_from_val}; extra in val: {extra_in_val}"
            )
