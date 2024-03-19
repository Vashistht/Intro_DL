
import unittest
import pickle as pk

from numpy.testing import assert_allclose
import torch

import deliverable2


TOLERANCE = 1e-4

with open('bleu_test.pk', "rb") as f: TESTS = pk.load(f)

class TestBleuScore(unittest.TestCase):

    def setUp(self):
        self.ans_key = "bleu_score"

    def test(self):
        target = [10, 11 , 4, 5, 6, 7, 8, 9]
        predicted = [10, 11, 4, 5, 6, 13, 12]

        for n in range(1, 5):
            assert_allclose(deliverable2.bleu_score(predicted, target, N=n), TESTS[self.ans_key][n-1], atol=TOLERANCE)

        predicted = [4, 4, 4, 4, 4, 4, 4, 4]
        assert_allclose(deliverable2.bleu_score(predicted, target, N=1), TESTS[self.ans_key][4], atol=TOLERANCE)

        predicted = [5, 15, 4, 10, 6]
        for n in range(1, 5):
            assert_allclose(deliverable2.bleu_score(predicted, target, N=n), TESTS[self.ans_key][n+4], atol=TOLERANCE)

        predicted = [4, 88, 4, 5, 6, 10, 11, 12, 7, 8]
        for n in range(1, 5):
            assert_allclose(deliverable2.bleu_score(predicted, target, N=n), TESTS[self.ans_key][n+8], atol=TOLERANCE)

        target = [99, 92, 6, 4, 4, 4, 5, 5, 5, 5]
        predicted = [6, 6, 6, 6, 4, 5]

        for n in range(1, 5):
            assert_allclose(deliverable2.bleu_score(predicted, target, N=n), TESTS[self.ans_key][n+12], atol=TOLERANCE)

        predicted = [11, 4]
        assert_allclose(deliverable2.bleu_score(predicted, target, N=4), TESTS[self.ans_key][17], atol=TOLERANCE)

        predicted = [12, 13, 4, 5, 6, 7]
        target = [10, 311]
        assert_allclose(deliverable2.bleu_score(predicted, target, N=4), TESTS[self.ans_key][18], atol=TOLERANCE)

test = TestBleuScore()
test.setUp()
test.test()
