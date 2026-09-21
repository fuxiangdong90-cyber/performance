import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
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
    def test_composite_addbmm_forward_and_random_vjp(self):
        from opbench.runner import addbmm_composite
        torch.manual_seed(37)
        c=torch.randn(3,5,dtype=torch.float64,requires_grad=True)
        a=torch.randn(4,3,7,dtype=torch.float64,requires_grad=True)
        b=torch.randn(4,7,5,dtype=torch.float64,requires_grad=True)
        v=torch.randn(3,5,dtype=torch.float64)
        actual=addbmm_composite(torch,c,a,b)
        expected=torch.addbmm(c,a,b)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        got=torch.autograd.grad(actual,(c,a,b),v)
        wanted=torch.autograd.grad(expected,(c,a,b),v)
        for x,y in zip(got,wanted):torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)

    def test_composite_low_precision_accumulation(self):
        from opbench.runner import addbmm_composite
        torch.manual_seed(2026)
        for dtype in (torch.float16, torch.bfloat16):
            a=torch.randn(8,32,64,dtype=dtype)
            b=torch.randn(8,64,32,dtype=dtype)
            c=torch.randn(32,32,dtype=dtype)
            expected=torch.addbmm(c.double(),a.double(),b.double()).to(dtype)
            torch.testing.assert_close(addbmm_composite(torch,c,a,b),expected,rtol=.01,atol=.01)

    def test_high_precision_reference_keeps_quantization_and_rms_epsilon(self):
        from opbench.runner import make_operation
        cases=expand_template({'schema_version':1,'operators':[
            {'operator':'RMSNorm','params':[{'shape':[2,4]}],'dtypes':['bfloat16']},
            {'operator':'LayerNorm','params':[{'shape':[8,256,64]}],'dtypes':['bfloat16'],'stages':['backward']}]})
        torch.manual_seed(17); state=torch.random.get_rng_state()
        rms,_=make_operation(torch,cases[0],'cpu',reference_dtype='float64')
        torch.random.set_rng_state(state)
        x=torch.randn(2,4,dtype=torch.bfloat16).double()
        expected=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+torch.finfo(torch.bfloat16).eps)
        torch.testing.assert_close(rms(),expected)
        op,_=make_operation(torch,cases[1],'cpu',reference_dtype='float64')
        torch.testing.assert_close(op()[-1],torch.full((64,),2048,dtype=torch.float64))

    def test_reference_rejects_wrong_output_and_restores_rng(self):
        from opbench import runner
        case=expand_template({'schema_version':1,'operators':[{'operator':'mm','params':[{'m':3,'n':4,'k':5}]}]})[0]
        original=runner.make_operation
        calls=[]
        def corrupt(*args,**kwargs):
            op,reset=original(*args,**kwargs); calls.append(1)
            return (lambda:op()+10,reset) if len(calls)==2 else (op,reset)
        state=torch.random.get_rng_state()
        with patch.object(runner,'make_operation',side_effect=corrupt):
            with self.assertRaises(AssertionError):runner.verify_reference(torch,case,'cpu')
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))

    def test_reference_checks_composite_and_optimizer(self):
        from opbench.runner import verify_reference
        cases=expand_template({'schema_version':1,'operators':[
            {'operator':'addbmm','implementation':'bmm_fp32_sum_v1','params':[{'batch':2,'m':3,'n':4,'k':5}],'stages':['forward','backward']},
            {'operator':'adamw','params':[{'elements':32}],'stages':['optimizer']}]})
        for case in cases:
            with self.subTest(op=case['operator'],stage=case['stage']):
                self.assertLessEqual(verify_reference(torch,case,'cpu')['max_scaled_error'],1)

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
