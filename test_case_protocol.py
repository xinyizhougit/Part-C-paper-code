import unittest
import tempfile
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
from data_processing import export_contract
from parameters import Config
from experiments import paired_days


class ProtocolTests(unittest.TestCase):
    def test_canonical_split_and_zero_targets(self):
        with tempfile.TemporaryDirectory() as root:
            c = replace(Config(), history=12, output=root)
            times = pd.date_range('2023-09-01', periods=200, freq='15min')
            raw = np.zeros((200, 3)); observed = np.ones_like(raw, bool)
            export_contract(raw, observed, times, ['a', 'b', 'c'], (140, 170), c)
            for name, lo, hi in [('train', 0, 140), ('val', 140, 170), ('test', 170, 200)]:
                e = np.load(Path(root, f'{name}_evaluation_contract.npz'))
                self.assertTrue(np.all(e['origins']+1 >= lo))
                self.assertTrue(np.all(e['origins']+4 < hi))
                self.assertTrue(e['mask'].all())
                self.assertEqual(e['y_true'].shape[1:], (3, 4))
                self.assertGreaterEqual(e['origins'].min()-c.history+1, 0)

    def test_paired_rejects_mismatched_labels(self):
        with tempfile.TemporaryDirectory() as root:
            a = dict(y_true=np.ones((3, 2, 4)), y_pred=np.zeros((3, 2, 4)),
                     mask=np.ones((3, 2, 4), bool), origins=np.arange(3), nodes=np.array(['a', 'b']),
                     target_times=np.array([[f'2023-10-0{i+1} 00:00:00']*4 for i in range(3)]))
            np.savez(Path(root, 'a.npz'), **a)
            a['y_true'] = a['y_true']+1
            np.savez(Path(root, 'b.npz'), **a)
            with self.assertRaisesRegex(ValueError, 'identical y_true'):
                paired_days(Path(root, 'a.npz'), Path(root, 'b.npz'), root)


if __name__ == '__main__':
    unittest.main()
