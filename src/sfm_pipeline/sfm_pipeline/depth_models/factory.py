from sfm_pipeline.depth_models.zoedepth_model import ZoeDepthModel
from sfm_pipeline.depth_models.depth_anything_v2_model import DepthAnythingV2MetricModel
from sfm_pipeline.depth_models.depth_pro_model import DepthProModel


def create_depth_model(model_name, device, logger=None, **kwargs):
    name = model_name.lower().strip()

    if name in ["zoedepth", "zoe", "zoed_n"]:
        return ZoeDepthModel(
            device=device,
            logger=logger
        )

    if name in [
        "depth_anything",
        "depth_anything_v2",
        "depth_anything_v2_metric",
        "dav2",
        "dav2_metric",
    ]:
        return DepthAnythingV2MetricModel(
            device=device,
            checkpoint_path=kwargs.get("checkpoint_path", ""),
            encoder=kwargs.get("encoder", "vitb"),
            max_depth=kwargs.get("max_depth", 20.0),
            logger=logger,
        )

    if name in ["depth_pro", "depthpro", "apple_depth_pro"]:
        return DepthProModel(
            device=device,
            logger=logger,
            input_size=kwargs.get("depth_pro_input_size", 512),
            force_cpu=kwargs.get("depth_pro_force_cpu", False),
        )

    raise ValueError(
        f"Unknown depth model: {model_name}. "
        "Available: zoedepth, depth_anything_v2_metric, depth_pro"
    )