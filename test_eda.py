"""Test the descriptive analysis using a small CSV and the full dataset."""

from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from eda import MISSING, rank_dot, summarize

ROOT = Path(__file__).resolve().parent


class TestEDA(unittest.TestCase):
    """Check data cleaning and summary statistics."""

    def test_load_and_summarize(self):
        """Compare the results with values calculated by hand."""
        data = pd.read_csv('test_students.csv', sep=';', na_values=MISSING)
        missing, quantitative, counts = summarize(data)
        self.assertEqual(missing['extra'], 2)
        self.assertEqual(missing['G3'], 0)
        self.assertEqual(data.loc[0, 'G3'], 0)
        expected = [10, 10, 0, 5, 10, 15, 20]
        for actual, value in zip(quantitative.loc['G3'], expected):
            self.assertAlmostEqual(actual, value)
        self.assertEqual(counts['studytime'], {1: 2, 2: 0, 3: 0, 4: 1})
        self.assertEqual(counts['Medu'][0], 1)
        with TemporaryDirectory() as folder:
            invalid_csv = Path(folder) / 'student-mat.csv'
            invalid_csv.write_text(
                'G3;absences;age;studytime;Medu;Fedu;school\n'
                '0;0;16;5;0;4;GP\n'
            )
            result = subprocess.run(
                [sys.executable, str(ROOT / 'eda.py')], cwd=folder,
                check=True, capture_output=True, text=True
            )
            self.assertIn('Invalid category in studytime: 5', result.stdout)
            self.assertFalse((Path(folder) / 'missing.csv').exists())

    def test_full_dataset(self):
        """Check the data shape and category totals."""
        data = pd.read_csv('student-mat.csv', sep=';', na_values=MISSING)
        missing, _, counts = summarize(data)
        self.assertEqual(data.shape, (395, 33))
        self.assertEqual(missing.sum(), 0)
        for category in counts:
            self.assertEqual(sum(counts[category].values()), 395)


class TestChallengeAnalysis(unittest.TestCase):
    """Check the rank statistic and run the full challenge analysis."""

    def test_rank_correlation(self):
        """Match SciPy's reference with tied observations and batching."""
        x = np.array([1, 1, 2, 3, 3])
        y = np.array([1, 2, 4, 3, 5])
        rho = spearmanr(x, y).statistic
        self.assertAlmostEqual(rank_dot(x, y), rho)
        np.testing.assert_allclose(rank_dot(np.array([x, -x]), y),
                                   [rho, -rho])

    def test_full_dataset(self):
        """Check the challenge outputs for both grade samples."""
        with TemporaryDirectory() as folder:
            shutil.copy(ROOT / 'student-mat.csv', folder)
            result = subprocess.run(
                [sys.executable, str(ROOT / 'eda.py')], cwd=folder,
                check=True, capture_output=True, text=True
            )
            output = Path(folder) / 'results'
            self.assertIn('Design rank: 15 of 15', result.stdout)
            self.assertGreater((output / 'diagnostics.png').stat().st_size, 0)
            pairs = pd.read_csv(output / 'rank_associations.csv')
            self.assertEqual(set(pairs.loc[pairs.school.eq('All'), 'n']),
                             {395, 357})


if __name__ == '__main__':
    unittest.main()
