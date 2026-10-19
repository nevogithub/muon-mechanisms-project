from .resnet import ResNet18
from .timm_tiny_vit import build_timm_tiny_vit

MODEL_REGISTRY = {
    "resnet18": ResNet18,
    "timm_tiny_vit": build_timm_tiny_vit,
}


def build_model(cfg):
    """
    Build model according to cfg["model"].
    """

    model_cfg = cfg["model"]
    model_name = model_cfg["name"]

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    model_class = MODEL_REGISTRY[model_name]

    if model_name == "timm_tiny_vit":
        return model_class(
            variant=model_cfg.get("variant", "tiny_vit_5m_224"),
            num_classes=model_cfg.get("num_classes", 100),
            pretrained=model_cfg.get("pretrained", False),
            drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
            expected_parameter_count=model_cfg.get("expected_parameter_count"),
        )

    if model_name == "resnet18":
        return model_class(num_classes=model_cfg["num_classes"])

    raise AssertionError(f"Unhandled model: {model_name}")

