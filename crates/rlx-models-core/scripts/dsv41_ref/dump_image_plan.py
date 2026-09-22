"""Dump `plan_image_grid` and the image-span token layout for the Rust port to match.

The resize plan is a pure function of the original pixel size and five config
values, and the span layout is a pure function of the token grid — so both can be
pinned exactly, without an image or a checkpoint.
"""

import json
import sys
from dataclasses import dataclass

from image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START, image_token_types, num_image_tokens, plan_image_grid


@dataclass
class Args:
    vision_patch_size: int
    vision_downsample_ratio: int
    vision_max_n_token: int
    vision_min_pixels: int
    vision_max_wh_ratio: float | None


def main() -> None:
    cfg = json.load(open("config.json"))
    v = cfg.get("vision_config", cfg)
    released = Args(
        vision_patch_size=v.get("patch_size", 14),
        vision_downsample_ratio=v.get("downsample_ratio", 3),
        vision_max_n_token=v.get("max_image_tokens", 1024),
        vision_min_pixels=v.get("min_pixels", 295936),
        vision_max_wh_ratio=v.get("max_wh_ratio"),
    )
    # a second, toy geometry so the Rust tests can use small numbers too
    toy = Args(2, 2, 64, 16, None)
    # and one with an aspect cap, which is its own branch
    capped = Args(14, 3, 1024, 295936, 4.0)

    cases = []
    for name, a in [("released", released), ("toy", toy), ("capped", capped)]:
        for w, h in [
            (64, 64), (1, 1), (4000, 30), (30, 4000), (1920, 1080), (1080, 1920),
            (100, 100), (7, 13), (2048, 2048), (333, 777), (16, 4000), (4000, 16),
            # Sizes where the `min_pixels` rescale lands off a patch boundary, so
            # truncating and rounding disagree. The first few shift only the
            # pixel size; (1, 162) changes the token grid itself.
            (1, 106), (1, 113), (1, 162), (1, 253), (3, 47), (5, 91),
        ]:
            n_llm_h, n_llm_w, best_h, best_w = plan_image_grid(w, h, a)
            cases.append({
                "args": name,
                "width": w,
                "height": h,
                "n_llm_h": n_llm_h,
                "n_llm_w": n_llm_w,
                "best_height": best_h,
                "best_width": best_w,
                "n_vit_h": best_h // a.vision_patch_size,
                "n_vit_w": best_w // a.vision_patch_size,
                "n_image_tokens": num_image_tokens(n_llm_h, n_llm_w),
            })

    layouts = []
    for h, w in [(1, 1), (1, 3), (3, 1), (2, 4), (5, 5)]:
        layouts.append({
            "n_llm_h": h,
            "n_llm_w": w,
            "types": image_token_types(h, w).tolist(),
        })

    out = {
        "type_ids": {
            "IMAGE_START": IMAGE_START,
            "IMAGE": IMAGE,
            "IMAGE_NEW_LINE": IMAGE_NEW_LINE,
            "IMAGE_END": IMAGE_END,
        },
        "args": {
            name: {
                "patch_size": a.vision_patch_size,
                "downsample_ratio": a.vision_downsample_ratio,
                "max_n_token": a.vision_max_n_token,
                "min_pixels": a.vision_min_pixels,
                "max_wh_ratio": a.vision_max_wh_ratio,
            }
            for name, a in [("released", released), ("toy", toy), ("capped", capped)]
        },
        "cases": cases,
        "layouts": layouts,
    }
    path = sys.argv[1] if len(sys.argv) > 1 else "../../tests/fixtures/dsv41_image_plan.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {len(cases)} plan cases and {len(layouts)} layouts to {path}")


if __name__ == "__main__":
    main()
