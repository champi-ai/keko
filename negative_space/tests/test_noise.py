import numpy as np
import pytest
import torch

from negative_space.noise import (add_relative_noise, flip_level, inject_noise, stability_auc,
                                  validate_schedule)


def test_schedule_validation():
    assert validate_schedule([0, 0.1, 0.2]) == [0.0, 0.1, 0.2]
    with pytest.raises(ValueError):
        validate_schedule([0.1, 0.2])
    with pytest.raises(ValueError):
        validate_schedule([0.0, 0.2, 0.2])
    with pytest.raises(ValueError):
        validate_schedule([0.0, 0.3, 0.1])


def test_relative_noise_scales_with_activation_norm():
    g = torch.Generator().manual_seed(0)
    for scale in (0.01, 1.0, 100.0):
        h = torch.randn(4, 50, 512, generator=g) * scale
        prev = 0.0
        for sigma in (0.05, 0.1, 0.2, 0.4):   # monotone in sigma, invariant to scale
            n = add_relative_noise(h, sigma, torch.Generator().manual_seed(1)) - h
            ratio = (n.norm(dim=-1) / h.norm(dim=-1)).mean().item()
            assert abs(ratio - sigma) < 0.05 * sigma + 1e-3
            assert ratio > prev
            prev = ratio
    assert torch.equal(add_relative_noise(h, 0.0, None), h)


def test_flip_level_and_auc():
    lv = [0.0, 0.1, 0.2, 0.4]
    assert flip_level(lv, np.array([1, 1, 0, 0]), 0.5) == (0.2, False)
    assert flip_level(lv, np.array([1, 1, 1, 1]), 0.5) == (0.8, True)
    assert stability_auc(lv, np.ones(4)) == pytest.approx(1.0)
    assert stability_auc(lv, np.array([1, 0, 0, 0])) < 0.2


@pytest.mark.parametrize("site", ["embedding", "hidden"])
def test_hook_perturbs_and_is_removed(tiny_model, site):
    model, tok = tiny_model
    ids = tok(["what is the capital of france ?"], return_tensors="pt")
    with torch.no_grad():
        clean = model(**ids).logits
        with inject_noise(model, site, 0.5, seed=0, hidden_layer=1):
            noisy = model(**ids).logits
        again = model(**ids).logits
    assert not torch.allclose(clean, noisy)
    assert torch.allclose(clean, again)
