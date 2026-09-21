import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from opbench.catalog import expand_template

try:
    import torch
except ImportError:
    torch=None


class PrecisionTests(unittest.TestCase):
    def test_vendor_tf32_is_disabled_and_reported_without_cuda(self):
        from opbench.runner import configure_precision
        fake = SimpleNamespace(backends=SimpleNamespace(mudnn=SimpleNamespace(allow_tf32=True)))
        fake.set_float32_matmul_precision = lambda value: setattr(fake, 'precision', value)
        fake.get_float32_matmul_precision = lambda: fake.precision
        result = configure_precision(fake)
        self.assertFalse(fake.backends.mudnn.allow_tf32)
        self.assertEqual(result, {'float32_matmul_precision':'highest', 'mudnn.allow_tf32':False})


@unittest.skipIf(torch is None,"install PyTorch to run real operator tests")
class RunnerTests(unittest.TestCase):
    def test_all_operators_forward_backward_optimizer_cpu(self):
        from opbench.runner import measure
        torch.set_num_threads(1)
        cases=expand_template(json.loads(Path('templates/smoke.json').read_text(encoding='utf-8')))
        for case in cases:
            with self.subTest(op=case['operator'],stage=case['stage']):
                record=measure(torch,case,'cpu',1,2)
                self.assertEqual(record['status'],'pass')
                self.assertGreater(record['wall_us'],0)
                self.assertEqual(len(record['samples_us']),2)
                self.assertIsNone(record['gpu_us'])
                self.assertIsNone(record['peak_allocated_bytes'])

    def test_backward_has_no_accumulation(self):
        from opbench.runner import make_operation
        case=expand_template({'schema_version':1,'operators':[{'operator':'mm','params':[{'m':3,'n':4,'k':5}],'stages':['backward']}]})[0]
        op,_=make_operation(torch,case,'cpu')
        first,second=op(),op()
        for a,b in zip(first,second):torch.testing.assert_close(a,b)

    def test_optimizer_reset_reproducible(self):
        from opbench.runner import make_operation
        for name in ['sgd','adam','adamw','adagrad','rmsprop']:
            case=expand_template({'schema_version':1,'operators':[{'operator':name,'params':[{'elements':16}],'stages':['optimizer']}]})[0]
            op,reset=make_operation(torch,case,'cpu')
            reset();first=op().detach().clone()
            reset();second=op().detach().clone()
            torch.testing.assert_close(first,second)


if __name__=='__main__':unittest.main()
