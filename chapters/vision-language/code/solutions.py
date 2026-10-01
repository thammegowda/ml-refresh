import numpy as np

from vlm import clip_zero_shot, dynamic_token_counts, mrope_position_ids


# tag::zero-shot[]
def toy_zero_shot_label():
    images = np.array([[1.0, 0.0], [0.0, 1.0]])
    prompts = np.array([[0.9, 0.1], [0.1, 0.9], [-1.0, 0.0]])
    return np.argmax(clip_zero_shot(images, prompts), axis=1).tolist()
# end::zero-shot[]


# tag::dynamic-counts[]
def dynamic_counts_example():
    return dynamic_token_counts(336, 672, patch_size=14, merge=2)
# end::dynamic-counts[]


# tag::mrope-small[]
def small_mrope_ids():
    return mrope_position_ids(frames=2, grid_h=2, grid_w=3)
# end::mrope-small[]
