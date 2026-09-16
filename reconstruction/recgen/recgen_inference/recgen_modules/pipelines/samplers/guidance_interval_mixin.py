from typing import *


class GuidanceIntervalSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance with interval.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, cfg_strength, cfg_interval, **kwargs):
        if cfg_interval[0] <= t <= cfg_interval[1]:
            result = super()._inference_model(model, x_t, t, cond, **kwargs)
            neg_result = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            
            # Handle both single and dual output (with pose)
            if isinstance(result, tuple) and len(result) == 2:
                pred, pred_pose = result
                neg_pred, neg_pred_pose = neg_result
                guided_pred = (1 + cfg_strength) * pred - cfg_strength * neg_pred
                # Apply guidance to pose as well if present
                if pred_pose is not None and neg_pred_pose is not None:
                    guided_pred_pose = (1 + cfg_strength) * pred_pose - cfg_strength * neg_pred_pose
                else:
                    guided_pred_pose = pred_pose
                return guided_pred, guided_pred_pose
            else:
                # Single output case
                return (1 + cfg_strength) * result - cfg_strength * neg_result
        else:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
