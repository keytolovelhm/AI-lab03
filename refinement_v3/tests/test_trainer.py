import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from refinement_v3 import train as trainer, mixing
from refinement_v3.config import BASE
from refinement_v3.corrected import shared_cap, evaluate
from refinement.training import base_summary
from refinement.core import batch_augmentation


def streams(seed):
    return {name: torch.Generator().manual_seed(seed + offset)
            for name, offset in trainer.STREAM_OFFSETS.items()}


def dataset(count=45000):
    labels = torch.arange(count)
    # Expanded immutable ID planes make crops/flips invariant without allocating 550 MB.
    images = labels.float()[:, None, None, None].expand(count, 32, 32, 3)
    return SimpleNamespace(train_images=images, train_labels=labels, normalization={})


class OrderTests(unittest.TestCase):
    def collect(self, seed, run=None, generators=None):
        data = dataset()
        batches = list(trainer.epoch_batches(data, run or {"crop": False, "flip": False},
                                            256, generators or streams(seed)))
        ids = torch.cat([positions for _, _, positions in batches])
        return batches, ids

    def test_complete_epoch_and_matching_labels(self):
        batches, ids = self.collect(11)
        self.assertTrue(torch.equal(ids.sort().values, torch.arange(45000)))
        for images, labels, positions in batches:
            self.assertTrue(torch.equal(labels, positions))
            self.assertTrue(torch.equal(images[:, 0, 0, 0], labels.float()))

    def test_seed_reproducible(self):
        self.assertTrue(torch.equal(self.collect(11)[1], self.collect(11)[1]))

    def test_epochs_and_seeds_differ(self):
        rng = streams(11)
        one = self.collect(11, generators=rng)[1]
        two = self.collect(11, generators=rng)[1]
        self.assertFalse(torch.equal(one, two))
        self.assertFalse(torch.equal(one, self.collect(12)[1]))

    def test_augmentation_keeps_sample_ids(self):
        for crop, flip in ((True, False), (False, True), (True, True)):
            for x, y, indices in trainer.epoch_batches(dataset(513), {"crop": crop, "flip": flip}, 128, streams(9)):
                self.assertTrue(torch.equal(x[:, 0, 0, 0], y.float()))
                self.assertTrue(torch.equal(y, indices))

    def test_stage_two_order_and_augmentation_equivalence(self):
        data = dataset(33)
        data.train_images = torch.randn(33, 32, 32, 3)
        rng = streams(7)
        x, labels, positions = next(trainer.epoch_batches(data, {"crop": True, "flip": True}, 16, rng))
        other = streams(7)
        expected_ids = torch.randperm(33, generator=other["order"])[:16]
        expected = batch_augmentation(data.train_images[expected_ids], other["crop"], other["flip"])
        self.assertTrue(torch.equal(positions, expected_ids))
        self.assertTrue(torch.equal(x, expected.permute(0, 3, 1, 2)))

    def test_mix_rng_independent_from_dropout(self):
        images = torch.ones(4, 3, 2, 2)
        before = torch.get_rng_state().clone()
        a = mixing.mixup_batch(images, .2, streams(3)["mix"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        b = mixing.mixup_batch(images, .2, streams(3)["mix"])
        self.assertEqual(a[2], b[2])
        self.assertTrue(torch.equal(a[1], b[1]))

    def test_cutmix_rng_independent_from_dropout(self):
        before = torch.get_rng_state().clone()
        images = torch.ones(4, 3, 32, 32)
        a = mixing.cutmix_batch(images, 1., streams(3)["mix"])
        b = mixing.cutmix_batch(images, 1., streams(3)["mix"])
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(a[2:], b[2:])
        self.assertTrue(torch.equal(a[1], b[1]))

    def test_training_loader_never_opens_test_file(self):
        from refinement_v3 import data
        original = data._load_cifar_batch
        seen = []
        def guarded(path):
            self.assertNotEqual(path.name, 'test_batch')
            seen.append(path.name)
            return original(path)
        data._load.cache_clear()
        with patch.object(data, '_load_cifar_batch', side_effect=guarded):
            loaded = data.load_training_dataset()
        self.assertEqual(len(loaded.train_labels), 45000)
        self.assertEqual(len(loaded.val_labels), 5000)
        self.assertEqual(len(loaded.test_labels), 0)
        self.assertEqual(seen, [f'data_batch_{i}' for i in range(1, 6)])
        data._load.cache_clear()


class BudgetTests(unittest.TestCase):
    def state(self, best_epoch=59, delta=0):
        history = [{"val_loss": 1.0} for _ in range(60)]
        for r in history[-6:]:
            r["val_loss"] += delta
        return {"history": history, "best": {"epoch": best_epoch, "val_accuracy": .8}, "stale": 1}

    def trigger(self, state):
        return base_summary(state, BASE, {"name": "fixture"}, 1)

    def test_best_last_epoch_triggers(self):
        self.assertTrue(self.trigger(self.state(60))["trigger"])

    def test_loss_threshold_inclusive(self):
        self.assertTrue(self.trigger(self.state(delta=-.006))["trigger"])
        self.assertFalse(self.trigger(self.state(delta=-.004))["trigger"])

    def test_early_stopped_does_not_trigger(self):
        state = self.state(28, -.1)
        state["history"] = state["history"][:28]
        self.assertFalse(self.trigger(state)["trigger"])

    def test_shared_any_trigger(self):
        self.assertEqual(shared_cap([{ "trigger": False}, {"trigger": True}]), 80)
        self.assertEqual(shared_cap([{ "trigger": False}]), 60)

    def test_sixty_epoch_stop_has_priority_over_extension(self):
        state = self.state(48, -.1)
        state['stale'] = 12
        # This special case can extend the shared cap while this run stays stopped.
        self.assertTrue(self.trigger(state)['trigger'])

    def test_full_cohort_precedes_shared_finalization(self):
        from refinement_v3 import corrected
        calls = []
        def fake(run, seed, **kwargs):
            calls.append((kwargs['budget'], kwargs.get('finalize', False)))
            if kwargs.get('finalize'):
                from refinement_v3.common import write_json
                write_json(kwargs['folder']/'training.json', {})
            return self.state(60)
        with tempfile.TemporaryDirectory() as directory, patch.object(corrected, 'RUNS', Path(directory)), \
                patch.object(corrected, 'validate_split', return_value=BASE['split_sha256']), \
                patch.object(corrected, 'train_one', side_effect=fake), patch.object(corrected, 'seal') as sealed:
            corrected.train()
            self.assertEqual(calls[:21], [(60, False)] * 21)
            self.assertEqual(calls[21:], [(80, True)] * 21)
            sealed.assert_called_once()

    def test_single_seed_waits_for_global_decision(self):
        from refinement_v3 import corrected
        with tempfile.TemporaryDirectory() as directory, patch.object(corrected, 'RUNS', Path(directory)), \
                patch.object(corrected, 'validate_split', return_value=BASE['split_sha256']), \
                patch.object(corrected, 'train_one', return_value=self.state()) as fit:
            corrected.train('se_both', 20261105)
            self.assertEqual(fit.call_count, 1)
            self.assertFalse((corrected.output_root()/'budget_decision.json').exists())

    def test_test_gate_precedes_dataset_loading(self):
        with tempfile.TemporaryDirectory() as directory, patch('refinement_v3.corrected.RUNS', Path(directory)), \
                patch('refinement_v3.data.load_dataset') as loader:
            with self.assertRaises(FileNotFoundError):
                evaluate()
            loader.assert_not_called()


class TrainingTests(unittest.TestCase):
    def test_smoothing_matches_explicit_soft_target(self):
        logits = torch.randn(4, 10)
        labels = torch.arange(4)
        target = .95 * torch.nn.functional.one_hot(labels, 10) + .05 / 10
        self.assertTrue(torch.allclose(mixing.soft_target_loss(logits, labels, None, 1., .05),
                                       torch.nn.functional.cross_entropy(logits, target)))

    def test_eval_preserves_bn_statistics_and_disables_dropout(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.Dropout(.5),
                                    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 2))
        data = dataset(8)
        labels = data.train_labels % 2
        before = model[0].running_mean.clone()
        a = trainer.clean_metrics(model, data.train_images, labels, {**BASE, 'precision': 'bf16'}, torch.device('cpu'))
        b = trainer.clean_metrics(model, data.train_images, labels, {**BASE, 'precision': 'bf16'}, torch.device('cpu'))
        self.assertEqual(a, b)
        self.assertTrue(torch.equal(before, model[0].running_mean))
        self.assertFalse(model[1].training)
    def setup_run(self, root, metrics, budget, finalize=False, mixing_kind="none", patience=12):
        data = dataset(8)
        data.train_labels %= 2
        data.val_images = data.train_images[:4]
        data.val_labels = data.train_labels[:4]
        data.normalization = {"fixture": True}
        run = {"name": "fixture", "crop": True, "flip": True, "mixing": mixing_kind, "mix_alpha": .2}
        def build(run, config):
            return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Dropout(.2), torch.nn.Linear(3072, 2)), {}
        def metric(*args, **kwargs):
            return next(metrics)
        overrides = {"precision": "fp32", "cpu_threads": 2, "batch_size": 8, "patience": patience}
        with patch.object(trainer, 'RUNS', root), patch.object(trainer, 'load_training_dataset', return_value=data), \
                patch.object(trainer, 'build', side_effect=build), patch.object(trainer, 'clean_metrics', side_effect=metric):
            return trainer.train_one(run, 11, overrides, log=lambda _: None, budget=budget, finalize=finalize)

    def test_early_stop_ties_and_best_checkpoint(self):
        # Epoch 2 wins the accuracy tie through lower loss. Exact ties then count as stale.
        metrics = iter([(1., .5), (1., .6), (1., .5), (.9, .6), (1., .5), (.9, .6), (1., .5), (1., .5)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = self.setup_run(root, metrics, 10, True, patience=2)
            self.assertEqual(summary['epochs_completed'], 4)
            self.assertEqual(summary['best_epoch'], 2)
            ck = torch.load(root/'suite_corrected/experiments/fixture/11/best.pt', weights_only=True)
            self.assertEqual(ck['best_epoch'], 2)
            self.assertEqual(summary['best_val_loss'], .9)

    def test_saved_best_weights_equal_selected_epoch(self):
        captured = {}
        original = trainer.save_checkpoint
        def capture(path, state):
            if path.name == 'resume.pt' and len(state['history']) == 2:
                captured.update({k:v.clone() for k,v in state['model'].items()})
            original(path, state)
        metrics = iter([(1., .5), (1., .6), (1., .5), (.9, .6), (1., .5), (1.1, .5)])
        with tempfile.TemporaryDirectory() as directory, patch.object(trainer, 'save_checkpoint', side_effect=capture):
            root = Path(directory)
            self.setup_run(root, metrics, 3, True)
            ck = torch.load(root/'suite_corrected/experiments/fixture/11/best.pt', weights_only=True)
            self.assertTrue(all(torch.equal(value, ck['model'][key]) for key,value in captured.items()))

    def test_scheduler_reaches_floor_and_holds(self):
        metrics = iter([(1., .5), (1., .5)] * 62)
        with tempfile.TemporaryDirectory() as directory:
            state = self.setup_run(Path(directory), metrics, 62, patience=100)
            history = state['history']
            self.assertEqual(len(history), 62)
            self.assertAlmostEqual(history[60]['lr'], BASE['min_learning_rate'])
            self.assertEqual(history[60]['lr'], history[61]['lr'])

    def test_interruption_resume_exact_weights_history_and_rng(self):
        def metrics():
            return iter([(1., .5), (.9, .6)] * 4)
        with tempfile.TemporaryDirectory() as directory:
            one, two = Path(directory)/'one', Path(directory)/'two'
            self.setup_run(one, metrics(), 4, mixing_kind="mixup")
            original = trainer.save_checkpoint
            def interrupt(path, state):
                original(path, state)
                if len(state['history']) == 2:
                    raise InterruptedError()
            with patch.object(trainer, 'save_checkpoint', side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    self.setup_run(two, metrics(), 4, mixing_kind="mixup")
            self.setup_run(two, iter([(1., .5), (.9, .6)] * 2), 4, mixing_kind="mixup")
            a = torch.load(one/'suite_corrected/experiments/fixture/11/resume.pt', weights_only=True)
            b = torch.load(two/'suite_corrected/experiments/fixture/11/resume.pt', weights_only=True)
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key], b['model'][key]))
            self.assertTrue(torch.equal(a['torch_rng'], b['torch_rng']))
            for x, y in zip(a['history'], b['history']):
                self.assertEqual({k:v for k,v in x.items() if k!='seconds'}, {k:v for k,v in y.items() if k!='seconds'})

    def test_historical_output_rejected(self):
        with self.assertRaises(ValueError):
            trainer.train_one({"name": "fixture"}, 1, folder=trainer.RUNS/'suite/experiments/fixture/1')
