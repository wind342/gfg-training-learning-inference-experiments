"""Regression tests for the post-execution read-only prefix audit."""
import copy
import unittest
from .audit import prefix_comparison

class PrefixAuditTests(unittest.TestCase):
    def records(self):
        info=dict(alpha=1.,old_loss_weight=.2,current_loss_weight=.8,
            gate_reason='NATIVE_SAFE',router_mode='RECOVERY',feedback_batch_size=4,
            selected_feedback_groups=[0,1],selected_anchor_groups=[1],
            release_event=False,router_activation=False,
            history=dict(states=['DEGRADED'],levels=[2],capabilities=[.8],q10=[.123]),
            baseline=[1.],predicted=[[1.]],actual=[1.],current_loss=.5)
        return [dict(step=i+211,info=copy.deepcopy(info)) for i in range(60)]
    def test_equal(self):
        a=self.records();self.assertTrue(prefix_comparison(a,a)['floating_bitwise_equal'])
    def test_float_difference_reported_not_hidden(self):
        a=self.records();b=copy.deepcopy(a);b[0]['info']['current_loss']+=1e-6
        r=prefix_comparison(a,b);self.assertFalse(r['floating_bitwise_equal']);self.assertGreater(r['maximum_absolute_float_difference'],0)
    def test_alpha_difference_rejected(self):
        a=self.records();b=copy.deepcopy(a);b[0]['info']['alpha']=.5
        with self.assertRaises(AssertionError):prefix_comparison(a,b)
    def test_boundary_sign_difference_rejected(self):
        a=self.records();b=copy.deepcopy(a);b[0]['info']['actual']=[-1.]
        with self.assertRaises(AssertionError):prefix_comparison(a,b)
    def test_coalition_difference_rejected(self):
        a=self.records();b=copy.deepcopy(a);b[0]['info']['selected_feedback_groups']=[0,2]
        with self.assertRaises(AssertionError):prefix_comparison(a,b)

if __name__=='__main__':unittest.main(verbosity=2)
