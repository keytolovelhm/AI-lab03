import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from refinement_v3.models import build_stage_model, remap_legacy_state_dict
from refinement_v3.state_loader import load_by_shape
from refinement_v3.config import BASE
from analysis_code.canonical import infer, load_checkpoint, temperature_invariant, output_directory

class MaintenanceTests(unittest.TestCase):
    def model(self):
        return build_stage_model(BASE,{'name':'fixture'})

    def test_loader_uses_semantic_keys_even_when_order_changes(self):
        one,two=self.model(),self.model()
        state=dict(reversed(list(one.state_dict().items())))
        load_by_shape(two,state)
        self.assertTrue(all(torch.equal(v,two.state_dict()[k]) for k,v in one.state_dict().items()))

    def test_loader_rejects_same_shape_wrong_keys(self):
        state=self.model().state_dict();state['wrong.weight']=state.pop('classifier.3.weight')
        with self.assertRaises(RuntimeError):load_by_shape(self.model(),state)

    def test_loader_rejects_missing_keys(self):
        state=self.model().state_dict();state.pop('classifier.3.bias')
        with self.assertRaises(RuntimeError):load_by_shape(self.model(),state)

    def test_legacy_block_mapping(self):
        state={'features.0.transition.0.0.weight':torch.ones(1),
               'features.2.branch.0.0.weight':torch.ones(1)}
        out=remap_legacy_state_dict(state)
        self.assertIn('features.1.transition.0.0.weight',out)
        self.assertIn('features.3.branch.0.0.weight',out)

    def test_legacy_stem_mapping(self):
        out=remap_legacy_state_dict({'stem.0.weight':torch.ones(1),'features.0.transition.0.0.weight':torch.ones(1)})
        self.assertIn('features.0.0.weight',out)
        self.assertIn('features.1.transition.0.0.weight',out)

    def test_canonical_eval_does_not_update_batchnorm(self):
        model=self.model();before={k:v.clone() for k,v in model.state_dict().items()}
        x=torch.randn(7,32,32,3)
        a,b=infer(model,x),infer(model,x)
        self.assertTrue(torch.equal(a,b));self.assertFalse(model.training)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))

    def test_autocast_rejected(self):
        with torch.autocast('cpu',dtype=torch.bfloat16),self.assertRaises(ValueError):
            infer(self.model(),torch.randn(2,32,32,3))

    def test_temperature_argmax_tested_on_logits(self):
        x=torch.tensor([[1.,3.,2.],[-1.,0.,2.]])
        for t in [.05,.85,1.,4.415]:self.assertTrue(temperature_invariant(x,t))

    def test_invalid_temperature_rejected(self):
        for t in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):temperature_invariant(torch.ones(2,10),t)

    def test_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'out';output_directory(path);(path/'keep').write_text('evidence')
            with self.assertRaises(FileExistsError):output_directory(path)
            self.assertEqual((path/'keep').read_text(),'evidence')

    def test_checkpoint_identity_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture'/'1'/'best.pt';p.parent.mkdir(parents=True)
            torch.save({'config':BASE,'run':{'name':'fixture'},'seed':2,'model':self.model().state_dict()},p)
            with self.assertRaises(ValueError):load_checkpoint(p)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture'/'1'/'best.pt';p.parent.mkdir(parents=True)
            one=self.model();torch.save({'config':BASE,'run':{'name':'fixture'},'seed':1,'model':one.state_dict()},p)
            two,_=load_checkpoint(p)
            self.assertFalse(two.training)
            self.assertTrue(all(torch.equal(v,two.state_dict()[k]) for k,v in one.state_dict().items()))

    def test_new_output_root(self):
        from refinement_v3.corrected import output_root
        with patch.dict('os.environ',{'LAB3_SUITE':'suite_maintenance'}):
            self.assertEqual(output_root().name,'suite_maintenance')

    def test_output_traversal_rejected(self):
        from refinement_v3.corrected import output_root
        with patch.dict('os.environ',{'LAB3_SUITE':'../escape'}),self.assertRaises(ValueError):output_root()
