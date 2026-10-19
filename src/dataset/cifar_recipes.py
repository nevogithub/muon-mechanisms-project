import torchvision.transforms as transforms


CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 2),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 4),
}


def recipe_uses_rand(recipe: str) -> bool:
    return str(recipe).lower() in {"full", "no_mix"}


def recipe_uses_mix(recipe: str) -> bool:
    return str(recipe).lower() in {"full", "no_rand"}


def build_cifar_transforms(
    dataset_name: str,
    recipe: str = "base",
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,
    random_erasing_p: float = 0.25,
    extra_train_tensor_transforms=None,
    extra_test_tensor_transforms=None,
):
    mean, std, crop_padding = CIFAR_STATS[dataset_name]
    recipe = str(recipe).lower()
    if recipe not in {"base", "full", "no_rand", "no_mix"}:
        raise ValueError(f"Unsupported CIFAR recipe: {recipe}")

    train_ops = [
        transforms.RandomCrop(32, padding=crop_padding),
        transforms.RandomHorizontalFlip(),
    ]
    if recipe_uses_rand(recipe):
        train_ops.append(
            transforms.RandAugment(
                num_ops=int(randaugment_num_ops),
                magnitude=int(randaugment_magnitude),
            )
        )
    train_ops.extend([transforms.ToTensor()])
    if extra_train_tensor_transforms:
        train_ops.extend(extra_train_tensor_transforms)
    train_ops.append(transforms.Normalize(mean, std))
    if recipe_uses_rand(recipe) and float(random_erasing_p) > 0.0:
        train_ops.append(transforms.RandomErasing(p=float(random_erasing_p)))

    test_ops = [transforms.ToTensor()]
    if extra_test_tensor_transforms:
        test_ops.extend(extra_test_tensor_transforms)
    test_ops.append(transforms.Normalize(mean, std))

    return transforms.Compose(train_ops), transforms.Compose(test_ops)
