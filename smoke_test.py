from dataclasses import replace
import ast
import unittest
import numpy as np
from parameters import Config
from data_processing import Cleaner
from indicators import calculate_metrics


class Checks(unittest.TestCase):
    def test_metrics(self):
        m = calculate_metrics(np.array([0., 2., 4.]), np.array([1., 3., 3.]), np.array([1, 1, 1], bool))
        self.assertAlmostEqual(m['MAE'], 1.)
        self.assertAlmostEqual(m['WMAPE'], 50.)
        self.assertIsNone(calculate_metrics(np.zeros(2), np.ones(2), np.ones(2, bool))['WMAPE'])

    def test_cleaner_train_only(self):
        rng = np.random.default_rng(42)
        x = rng.poisson(3, (100, 6)).astype(float)
        x[0, 0] = np.nan
        cleaner = Cleaner().fit(x[:70], Config())
        before = cleaner.med.copy()
        x[70:, 0] = 100000
        z = cleaner.transform(x)
        np.testing.assert_array_equal(before, cleaner.med)
        self.assertTrue(np.isfinite(z).all())

    def test_model_shapes_and_gradient(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch unavailable')
        from model import DTSSM
        c = replace(Config(), history=4, hidden=8, rpca_iterations=3)
        model = DTSSM(c)
        x = torch.randn(2, 4, 5)
        relation = torch.rand(2, 4, 3, 5, 5)
        bnp = torch.ones(2, 4, 5, 5)
        support = torch.ones_like(bnp)
        selection = torch.ones(2, 4, c.max_hop, 5, 5)
        y = model(x, [relation, bnp, support, selection])
        self.assertEqual(tuple(y.shape), (2, 5, 4))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(model.gamma.grad).all())
        zero = model(x, [relation*0, bnp*0, support*0, selection*0])
        self.assertTrue(torch.isfinite(zero).all())


if __name__ == '__main__':
    unittest.main()
