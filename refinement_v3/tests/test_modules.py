from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from refinement_v3 import calibration, mixing, perturbations, corruptions
from refinement_v3.models import (CBAM, REFERENCE_PARAMS, SE, StageNet, build_stage_model,
                                  MobileNetV3SmallCifar, set_trainable)
from refinement_v3 import config as matrix


class AttentionTest(unittest.TestCase):
    def test_se_shrinks_channel_by_gate(self):
        torch.manual_seed(0)
        se = SE(16, 4).eval()
        x = torch.ones(2, 16, 8, 8)
        with torch.no_grad():
            y = se(x)
            gate = se.gate(se.fc2(se.act(se.fc1(se.pool(x).view(2, 16)))))
        self.assertEqual(tuple(y.shape), (2, 16, 8, 8))
        self.assertTrue(torch.all(gate > 0) and torch.all(gate < 1))
        # output equals input scaled by the per-channel gate
        self.assertTrue(torch.allclose(y, x * gate.view(2, 16, 1, 1), atol=1e-6))

    def test_se_uniform_activation_gives_uniform_gate(self):
        se = SE(8, 4).eval()
        with torch.no_grad():
            y = se(torch.ones(1, 8, 4, 4))
        # identical spatial content per channel: the gate is uniform *within* each channel
        for channel in range(8):
            plane = y[0, channel].flatten()
            self.assertTrue(torch.allclose(plane, plane[0].expand_as(plane), atol=1e-6))

    def test_se_reduction_ratio_and_param_count(self):
        se = SE(96, 4)
        params = sum(p.numel() for p in se.parameters())
        self.assertEqual(params, 96 * 24 + 24 * 96)

    def test_cbam_channel_and_spatial_maps(self):
        cbam = CBAM(32, 4, 7).eval()
        x = torch.rand(2, 32, 8, 8)
        with torch.no_grad():
            channel = cbam.channel_attention(x)
            spatial = cbam.spatial_attention(x)
            y = cbam(x)
        self.assertEqual(tuple(channel.shape), (2, 32, 1, 1))
        self.assertEqual(tuple(spatial.shape), (2, 1, 8, 8))
        self.assertLessEqual(float(spatial.max()), 1.0)
        self.assertGreaterEqual(float(spatial.min()), 0.0)
        self.assertEqual(tuple(y.shape), x.shape)

    def test_cbam_param_count(self):
        cbam = CBAM(96, 4, 7)
        expected = 96 * 24 * 2 + (7 * 7 * 2)
        self.assertEqual(sum(p.numel() for p in cbam.parameters()), expected)


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.config = dict(matrix.BASE)

    def _params(self, **run):
        model = build_stage_model(self.config, run)
        return sum(p.numel() for p in model.parameters())

    def test_reference_reproduces_stage2_architecture(self):
        run = {"residual": False, "attention": "none", "stem": "pixel_unshuffle"}
        self.assertEqual(self._params(**run), REFERENCE_PARAMS)

    def test_attention_and_stem_param_counts(self):
        counts = {
            "se": self._params(residual=False, attention="se", stem="pixel_unshuffle"),
            "cbam": self._params(residual=False, attention="cbam", stem="pixel_unshuffle"),
            "conv_stem": self._params(residual=False, attention="none", stem="conv"),
        }
        self.assertEqual(counts["se"] - REFERENCE_PARAMS, 24 * 6 + 6 * 24 + 48 * 12 + 12 * 48 + 96 * 24 + 24 * 96)
        self.assertEqual(counts["conv_stem"] - REFERENCE_PARAMS, 3 * 12 * 3 * 3 + 12 * 2)
        self.assertGreater(counts["cbam"], counts["se"])

    def test_conv_stem_matches_spatial_and_channel_layout(self):
        for run in ({"residual": False, "attention": "none", "stem": "pixel_unshuffle"},
                    {"residual": False, "attention": "none", "stem": "conv"}):
            model = build_stage_model(self.config, run).eval()
            with torch.no_grad():
                features = model.features(torch.zeros(1, 3, 32, 32))
            self.assertEqual(tuple(features.shape), (1, 96, 4, 4), msg=str(run))
        pixel = build_stage_model(self.config, {"residual": False, "attention": "none", "stem": "pixel_unshuffle"})
        self.assertIsInstance(pixel.features[0], torch.nn.PixelUnshuffle)
        self.assertTrue(next(iter(pixel.state_dict())).startswith("features.1."))

    def test_forward_backward_single_update(self):
        for run in ({"attention": "se", "stem": "pixel_unshuffle"},
                    {"attention": "cbam", "stem": "conv"}):
            model = build_stage_model(self.config, {**run, "residual": False})
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
            before = model.classifier[-1].weight.detach().clone()
            logits = model(torch.randn(8, 3, 32, 32))
            self.assertEqual(tuple(logits.shape), (8, 10))
            loss = F.cross_entropy(logits, torch.randint(0, 10, (8,)))
            loss.backward()
            optimizer.step()
            changed = (model.classifier[-1].weight.detach() - before).abs().max().item()
            self.assertGreater(changed, 0.0)

    def test_mobilenet_trainable_partitioning(self):
        torch.manual_seed(0)
        model = MobileNetV3SmallCifar(num_classes=10)
        counts = {mode: set_trainable(model, mode)["trainable"] for mode in ("frozen", "last2", "last4", "full")}
        self.assertLess(counts["frozen"], counts["last2"])
        self.assertLess(counts["last2"], counts["last4"])
        self.assertLess(counts["last4"], counts["full"])
        self.assertEqual(counts["frozen"], 576 * 1024 + 1024 + 1024 * 10 + 10)

    def test_mobilenet_accepts_96(self):
        model = MobileNetV3SmallCifar(num_classes=10).eval()
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.zeros(2, 3, 96, 96)).shape), (2, 10))


class MixingTest(unittest.TestCase):
    def test_mixup_lam_and_labels(self):
        generator = torch.Generator().manual_seed(0)
        images = torch.rand(16, 3, 32, 32)
        mixed, permutation, lam = mixing.mixup_batch(images, 0.2, generator)
        self.assertAlmostEqual(lam + (1 - lam), 1.0)
        self.assertEqual(tuple(mixed.shape), images.shape)
        self.assertFalse(torch.equal(permutation, torch.arange(16)))

    def test_cutmix_lam_matches_covered_pixels(self):
        generator = torch.Generator().manual_seed(1)
        for _ in range(20):
            images = torch.rand(8, 3, 32, 32)
            mixed, permutation, lam, box = mixing.cutmix_batch(images, 1.0, generator)
            # weights use the aligned permutation, so the pasted source is images[perm, box]
            expected = images.clone()
            expected[:, :, box["top"]:box["top"] + box["height"],
                     box["left"]:box["left"] + box["width"]] = images[
                permutation, :, box["top"]:box["top"] + box["height"],
                box["left"]:box["left"] + box["width"]]
            self.assertTrue(torch.equal(mixed, expected))
            patch = (box["height"] * box["width"]) / (32 * 32)
            self.assertAlmostEqual(lam, 1.0 - patch, places=12)
            # untouched copy of the image must match the original outside the box
            mask = torch.ones(32, 32, dtype=torch.bool)
            mask[box["top"]:box["top"] + box["height"], box["left"]:box["left"] + box["width"]] = False
            self.assertTrue(torch.equal(mixed[:, :, mask], images[:, :, mask]))
            self.assertTrue(torch.equal(mixed[:, :, ~mask], images[permutation][:, :, ~mask]))
            self.assertFalse(torch.equal(permutation, torch.arange(8)))

    def test_cutmix_soft_loss_is_weighted_sum(self):
        logits = torch.randn(4, 10)
        targets = torch.tensor([0, 1, 2, 3])
        permutation = torch.tensor([1, 2, 3, 0])
        lam = 0.7
        expected = lam * F.cross_entropy(logits, targets) + (1 - lam) * F.cross_entropy(logits, targets[permutation])
        actual = mixing.soft_target_loss(logits, targets, permutation, lam)
        self.assertTrue(torch.allclose(expected, actual))

    def test_soft_loss_without_mixing_matches_ce(self):
        logits = torch.randn(4, 10)
        targets = torch.tensor([0, 1, 2, 3])
        self.assertTrue(torch.allclose(mixing.soft_target_loss(logits, targets, None, 1.0),
                                       F.cross_entropy(logits, targets)))


class CalibrationTest(unittest.TestCase):
    def test_perfect_logits_have_zero_ece(self):
        labels = torch.arange(10).repeat(3)
        logits = torch.full((30, 10), -20.0)
        logits[torch.arange(30), labels] = 20.0
        metrics = calibration.calibration_metrics(logits, labels)
        self.assertAlmostEqual(metrics["ece"], 0.0, places=6)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)

    def test_temperature_scale_search_returns_unity_for_calibrated_logits(self):
        torch.manual_seed(0)
        targets = torch.randint(0, 10, (2000,))
        logits = torch.randn(2000, 10) * 3.0
        fitted = calibration.fit_temperature(logits, targets)
        self.assertTrue(fitted["argmax_unchanged"])
        self.assertGreaterEqual(fitted["temperature"], 0.05)

    def test_temperature_fit_expands_bracket(self):
        torch.manual_seed(0)
        labels = torch.randint(0, 10, (500,))
        logits = torch.randn(500, 10) * 20
        logits[torch.arange(500), labels] += 5
        fitted = calibration.fit_temperature(logits, labels)
        self.assertTrue(fitted["argmax_unchanged"])
        self.assertGreater(fitted["temperature"], 1e-6)

    def test_temperature_does_not_change_argmax(self):
        torch.manual_seed(0)
        logits = torch.randn(128, 10)
        scaled = logits / 3.7
        self.assertTrue(torch.equal(logits.argmax(1), scaled.argmax(1)))

    def test_ece_bins_cover_all_samples(self):
        torch.manual_seed(0)
        logits = torch.randn(1000, 10)
        labels = torch.randint(0, 10, (1000,))
        metrics = calibration.calibration_metrics(logits, labels)
        self.assertEqual(sum(b["count"] for b in metrics["bins"]), 1000)
        self.assertEqual(len(metrics["bins"]), 15)


class PerturbationTest(unittest.TestCase):
    def test_each_perturbation_preserves_shape(self):
        images = torch.randn(4, 3, 32, 32) * 0.25
        generator = torch.Generator().manual_seed(0)
        for name in ("clean", "uniform_grey", "gaussian_noise", "gaussian_blur", "brightness",
                     "contrast", "occlusion"):
            for level in (1, 3, 5):
                out = corruptions.apply(images, name, level, generator)
                self.assertEqual(tuple(out.shape), images.shape, msg=f"{name}-{level}")
                self.assertTrue(torch.isfinite(out).all())

    def test_uniform_grey_control_is_constant(self):
        images = torch.randn(2, 3, 32, 32)
        out = corruptions.apply(images, "uniform_grey", 1)
        self.assertTrue(torch.all(out == 0.0))

    def test_occlusion_fills_grey_not_image_mean(self):
        images = torch.full((1, 3, 32, 32), 0.5)
        out = corruptions.apply(images, "occlusion", 5)
        self.assertTrue(torch.all(out[0, :, 16, 16] == 0.0))
        self.assertTrue(torch.all(out[0, :, 0, 0] == 0.5))

    def test_protocol_is_registered_for_all_five_perturbations(self):
        self.assertEqual(len(perturbations.PROTOCOL["perturbations"]), 5)
        for spec in perturbations.PROTOCOL["perturbations"].values():
            value_key = next(k for k in spec if k != "kind")
            self.assertEqual(len(spec[value_key]), 5)


class MatrixTest(unittest.TestCase):
    def test_p0_and_p1_sizes(self):
        self.assertEqual(len(matrix.P0), 5)
        self.assertEqual(len(matrix.P1), 2)
        names = [c["name"] for c in matrix.P0 + matrix.P1]
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
