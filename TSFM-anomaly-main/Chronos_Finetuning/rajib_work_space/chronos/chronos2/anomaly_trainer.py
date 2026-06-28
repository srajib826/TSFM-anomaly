# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from chronos.chronos2.trainer import Chronos2Trainer


class Chronos2AnomalyTrainer(Chronos2Trainer):
    """
    Stage-2 trainer for anomaly detection fine-tuning.

    Negates the loss (gradient ascent) so the model learns to predict
    badly on futures that contain anomaly signals. At inference time,
    high prediction error on a region = high anomaly score.

    Parameters
    ----------
    loss_ceiling : float or None
        If set, the raw (positive) loss is clamped to this value before negation.
        This prevents gradient ascent from diverging: once the model predicts
        badly enough (loss >= ceiling), gradients become zero and training stops
        pushing further. Recommended: ~3-5x the final stage-1 eval loss.

    Usage:
        import functools
        stage2_pipeline = stage1_pipeline.fit(
            inputs=anomaly_future_dataset,
            prediction_length=pred_len,
            trainer_cls=functools.partial(Chronos2AnomalyTrainer, loss_ceiling=15.0),
        )
    """

    def __init__(self, *args, loss_ceiling=None, **kwargs):
        # print("-------------------Chronos2AnomalyTrainer----------------------")
        super().__init__(*args, **kwargs)
        self.loss_ceiling = loss_ceiling

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
       


        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        # Clamp before negating so ascent stops once loss is "bad enough".
        # Without this, the model can diverge to arbitrarily large losses.
        if self.loss_ceiling is not None:
            loss = loss.clamp(max=self.loss_ceiling)

        # Negate loss to perform gradient ascent:
        # minimizing -loss = maximizing loss = model learns to predict badly on anomaly futures
        loss = -loss

        return (loss, outputs) if return_outputs else loss
