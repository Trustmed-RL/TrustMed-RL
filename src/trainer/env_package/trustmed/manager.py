"""EnvironmentManager for trustmed (verl-agent fork side)."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Tuple

try:
    from agent_system.environments.base import EnvironmentManagerBase
except Exception:
    EnvironmentManagerBase = object


def _decode_parts(multi_modal) -> list:
    """OpenAI image_url parts (the env's chat format) -> raw PIL images (the fork's processor input). The env base64-encodes each served crop into a data URI; here they are decoded back so verl's image_processor can consume them."""
    if not multi_modal:
        return []
    from PIL import Image

    imgs = []
    for part in multi_modal:
        if not isinstance(part, dict):
            continue
        url = (part.get("image_url") or {}).get("url", "")
        if url.startswith("data:") and "," in url:
            try:
                raw = base64.b64decode(url.split(",", 1)[1])
                imgs.append(Image.open(io.BytesIO(raw)).convert("RGB"))
            except Exception:
                continue
    return imgs


class TrustMedEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)

    @staticmethod
    def _pack(obs_list: List[dict], infos: List[dict] | None = None) -> Dict[str, Any]:
        """per-env [{"text","multi_modal"}] -> {"text":[...],"image":[...],"anchor":[...]}."""
        texts: list[str] = []
        images: list = []
        for o in obs_list:
            txt = o.get("text", "") if isinstance(o, dict) else ""
            imgs = _decode_parts(o.get("multi_modal") if isinstance(o, dict) else None)
            if imgs:
                texts.append("<image>" * len(imgs) + txt)
                images.append(imgs)
            else:
                texts.append(txt)
                images.append(None)
        if infos is not None:
            anchors = [
                (inf.get("anchor_obs") or "reset") if isinstance(inf, dict) else "reset"
                for inf in infos
            ]
        else:
            anchors = ["reset"] * len(obs_list)
        has_image = any(im for im in images)
        return {"text": texts, "image": images if has_image else None, "anchor": anchors}

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs_list, infos = self.envs.reset(kwargs=kwargs)
        return self._pack(obs_list, infos=infos), infos

    def step(self, text_actions: List[str]):
        actions, _valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        return self._pack(next_obs, infos=infos), rewards, dones, infos

    def abort_one(self, i: int, reason: str = "prompt_overflow") -> dict:
        """The collector could not build env i's prompt (over data.max_prompt_length, truncation=error): end that episode without an action. Forwards to TrustMedMultiThreadEnv.abort_one; every later step() pads env i as done."""
        fn = getattr(self.envs, "abort_one", None)
        return fn(i, reason) if fn is not None else {}
