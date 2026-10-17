"""Versioned TT2 objective contracts; never reinterpret an existing run.

stabilized_seconds_v1 is an experimental practical recipe, not an assertion of
the duration units/initialization used by the unpublished original TT2 code.
"""
from dataclasses import replace
import math

LEGACY = 'legacy_frames'
STABILIZED = 'stabilized_seconds_v1'
POOLED = 'stabilized_pooled_v2'
STABILIZED_RECIPES = (STABILIZED, POOLED)
RECIPES = (LEGACY, *STABILIZED_RECIPES)
STABILIZED_LR = 0.0003
STABILIZED_WARMUP = 400


def validate_recipe(recipe, mode):
    if recipe not in RECIPES:
        raise ValueError('unknown TT2 recipe')
    if recipe in STABILIZED_RECIPES and mode != 'paper_practical':
        raise ValueError(f'{recipe} requires paper_practical and a new run')


def recipe_model(config, recipe, target_spec):
    if recipe == LEGACY:
        return config
    if recipe not in STABILIZED_RECIPES:
        raise ValueError('unknown TT2 recipe')
    hop, rate = target_spec.get('hop_length'), target_spec.get('sample_rate')
    if (type(hop) is not int or type(rate) is not int or min(hop, rate) <= 0
            or not math.isfinite(hop / rate) or hop >= rate):
        raise ValueError('seconds duration requires a valid target hop_length/sample_rate')
    return replace(config, duration_unit='seconds', duration_hop_seconds=hop / rate,
                   duration_output_scale=0.1, duration_initial_value=0.08)


def recipe_metadata(recipe):
    if recipe == LEGACY:
        return {}
    if recipe not in STABILIZED_RECIPES:
        raise ValueError('unknown TT2 recipe')
    result = dict(tt2_recipe=recipe, duration_unit='seconds',
                duration_objective='mean((sum(predicted_seconds)-target_frames*hop_seconds)^2)',
                duration_output_scale=0.1, duration_initial_value=0.08,
                upsampling_unit='frames', gaussian_range_unit='frames',
                schedule_contract='fixed peak LR and warmup; no resume-time shortening')
    if recipe == POOLED:
        result['evaluation_statistics'] = 'train-only pooled postnet; checkpoint-bound; native training buffers retained'
    return result
