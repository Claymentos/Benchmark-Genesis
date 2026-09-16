from typing import *


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, **kwargs):
        pred, pos_pose_pred = super()._inference_model(model, x_t, t, cond, **kwargs)
        neg_pred, neg_pose_pred = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        
        # Apply CFG to structure prediction
        cfg_pred = (1 + cfg_strength) * pred - cfg_strength * neg_pred
        
        # Apply CFG to pose prediction if present
        if pos_pose_pred is not None and neg_pose_pred is not None:
            cfg_pose_pred = (1 + cfg_strength) * pos_pose_pred - cfg_strength * neg_pose_pred
        else:
            cfg_pose_pred = None
        
        return cfg_pred, cfg_pose_pred
