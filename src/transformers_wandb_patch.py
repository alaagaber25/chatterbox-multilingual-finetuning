from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def patch_wandb_on_train_end_eval_strategy() -> None:
    try:
        from transformers.integrations.integration_utils import WandbCallback
        from transformers.trainer_utils import IntervalStrategy
    except Exception:
        return

    if getattr(WandbCallback, "_voom_eval_strategy_patch", False):
        return

    original_on_train_end = WandbCallback.on_train_end

    def patched_on_train_end(self, args, state, control, **kwargs):
        log_model = getattr(self, "_log_model", None)
        if not getattr(log_model, "is_enabled", False):
            return original_on_train_end(self, args, state, control, **kwargs)

        restore_eval_strategy = None
        eval_strategy = getattr(args, "eval_strategy", None)
        if eval_strategy not in (None, IntervalStrategy.NO, "no"):
            restore_eval_strategy = eval_strategy
            args.eval_strategy = IntervalStrategy.NO

        try:
            return original_on_train_end(self, args, state, control, **kwargs)
        finally:
            if restore_eval_strategy is not None:
                args.eval_strategy = restore_eval_strategy

    WandbCallback.on_train_end = patched_on_train_end
    WandbCallback._voom_eval_strategy_patch = True
    logger.info(
        "Patched WandbCallback.on_train_end for final artifact saving with eval_strategy enabled."
    )
