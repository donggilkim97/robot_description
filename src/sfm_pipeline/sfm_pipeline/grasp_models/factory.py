from sfm_pipeline.grasp_models.pca_grasp_model import PCAGraspModel


def create_grasp_model(model_name, logger=None, **kwargs):
    name = model_name.lower().strip()

    if name in ["pca", "baseline", "rule_based", "rule-based"]:
        if logger is not None:
            logger.info("Using PCA rule-based grasp model.")

        return PCAGraspModel(
            z_offset=kwargs.get("z_offset", 0.035)
        )

    if name in ["ggcnn", "gg-cnn", "ggcnn2", "gg-cnn2"]:
        raise NotImplementedError(
            "GGCNN grasp model wrapper is not implemented yet. "
            "The interface is ready, so it can be added later."
        )

    if name in ["vgn"]:
        raise NotImplementedError(
            "VGN grasp model wrapper is not implemented yet. "
            "The interface is ready, so it can be added later."
        )

    if name in ["contact_graspnet", "contact-graspnet"]:
        raise NotImplementedError(
            "Contact-GraspNet wrapper is not implemented yet. "
            "The interface is ready, so it can be added later."
        )

    raise ValueError(
        f"Unknown grasp model: {model_name}. "
        "Available now: pca"
    )