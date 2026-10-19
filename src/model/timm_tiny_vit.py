"""Factory for the reference TinyViT implementation shipped by timm."""


def build_timm_tiny_vit(
    variant="tiny_vit_5m_224",
    num_classes=100,
    pretrained=False,
    drop_path_rate=0.0,
    expected_parameter_count=None,
):
    try:
        import timm
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The reference TinyViT model requires timm. Install the ImageNet-100 "
            "dependencies with: pip install -r requirements-imagenet100.txt"
        ) from exc

    available = set(timm.list_models("tiny_vit_*"))
    if variant not in available:
        raise ValueError(
            f"Unknown timm TinyViT variant {variant!r}. Available variants: "
            f"{', '.join(sorted(available))}"
        )

    model = timm.create_model(
        variant,
        pretrained=bool(pretrained),
        num_classes=int(num_classes),
        drop_path_rate=float(drop_path_rate),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if (
        expected_parameter_count is not None
        and parameter_count != int(expected_parameter_count)
    ):
        raise ValueError(
            f"Unexpected {variant} parameter count: expected "
            f"{expected_parameter_count}, found {parameter_count}"
        )
    return model
