from sfm_pipeline.depth_models.zoedepth_model import ZoeDepthModel


def create_depth_model(model_name, device, logger=None):
    name = model_name.lower().strip()

    if name in ["zoedepth", "zoe", "zoed_n"]:
        return ZoeDepthModel(device=device, logger=logger)

    if name in ["depth_anything", "depth_anything_v2"]:
        raise NotImplementedError(
            "Depth Anything V2 wrapper is not implemented yet. "
            "The interface is ready, so it can be added later."
        )

    if name in ["depth_pro", "depthpro"]:
        raise NotImplementedError(
            "Depth Pro wrapper is not implemented yet. "
            "The interface is ready, so it can be added later."
        )

    raise ValueError(
        f"Unknown depth model: {model_name}. "
        "Available now: zoedepth"
    )