import torchvision

from .cifar_recipes import build_cifar_transforms


class CIFAR100:
    """
    A wrapper class for loading and preprocessing the CIFAR-100 dataset.

    This class defines training and testing datasets with specific data augmentation
    and normalization transformations applied to the images. It provides the datasets
    in a format ready for use with PyTorch DataLoaders.

    Attributes:
    - trainset (torchvision.datasets.CIFAR100): The training dataset with data augmentations.
    - testset (torchvision.datasets.CIFAR100): The testing dataset with normalization applied.
    - trainset_eval (torchvision.datasets.CIFAR100): Clean train set for evaluation (no augmentation).
    """

    def __init__(
        self,
        root="./data",
        recipe="base",
        randaugment_num_ops=2,
        randaugment_magnitude=9,
        random_erasing_p=0.25,
    ):
        """
        Initializes the CIFAR100 class.

        The class sets up training and testing datasets with appropriate transformations.
        Data augmentation is applied to the training dataset, while only normalization
        is applied to the testing dataset.
        """
        super().__init__()

        transform_train, transform_test = build_cifar_transforms(
            "cifar100",
            recipe=recipe,
            randaugment_num_ops=randaugment_num_ops,
            randaugment_magnitude=randaugment_magnitude,
            random_erasing_p=random_erasing_p,
        )

        # Load the training dataset with transformations
        self.trainset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform_train
        )

        # Clean train set for evaluation (no augmentation, same transforms as test)
        self.trainset_eval = torchvision.datasets.CIFAR100(
            root=root, train=True, download=False, transform=transform_test
        )

        # Load the testing dataset with transformations
        self.testset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True, transform=transform_test
        )

