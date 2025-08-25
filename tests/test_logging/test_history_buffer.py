# pylint: disable=invalid-name, inconsistent-quotes, unused-argument,
# pylint: disable=function-redefined, use-implicit-booleaness-not-comparison
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from pipeai.logging import HistoryBuffer


@HistoryBuffer.register_statistics
def custom_statistics(self):
    return -1


class TestHistoryBuffer(unittest.TestCase):

    def setUp(self):
        log_buffer = HistoryBuffer()
        assert log_buffer.max_length == 1000000

        log_history, counts = log_buffer.data
        assert len(log_history) == 0
        assert len(counts) == 0

        # test the length of array exceed `max_length`
        logs = np.random.randint(1, 10, log_buffer.max_length + 1)
        counts = np.random.randint(1, 10, log_buffer.max_length + 1)
        log_buffer = HistoryBuffer(logs, counts)
        log_history, count_history = log_buffer.data
        assert len(log_history) == log_buffer.max_length
        assert len(count_history) == log_buffer.max_length
        assert logs[1] == log_history[0]
        assert counts[1] == count_history[0]

        # The different lengths of `log_history` and `count_history` will
        # raise error
        with pytest.raises(AssertionError):
            HistoryBuffer([1, 2], [1])

    @patch("numpy.array", side_effect=lambda x: np.asarray(x))
    def test_update(self, mock_array):
        log_buffer = HistoryBuffer()
        log_history = mock_array([1, 2, 3, 4, 5])
        count_history = mock_array([5, 5, 5, 5, 5])
        for i in range(len(log_history)):
            log_buffer.update(float(log_history[i]), float(count_history[i]))
        self.assertTrue(mock_array.called)
        np.testing.assert_array_equal(log_buffer._log_history, np.array([1, 2, 3, 4, 5]))
        np.testing.assert_array_equal(log_buffer._count_history, np.array([5, 5, 5, 5, 5]))

        recorded_history, recorded_count = log_buffer.data
        for a, b in zip(log_history, recorded_history):
            assert float(a) == float(b)
        for a, b in zip(count_history, recorded_count):
            assert float(a) == float(b)

        # test the length of `array` exceed `max_length`
        max_array = mock_array([[-1] + [1] * (log_buffer.max_length - 1)])
        max_count = mock_array([[-1] + [1] * (log_buffer.max_length - 1)])
        log_buffer = HistoryBuffer(max_array, max_count)
        log_buffer.update(1)
        log_history, count_history = log_buffer.data
        assert log_history[0] == 1
        assert count_history[0] == 1
        assert len(log_history) == log_buffer.max_length
        assert len(count_history) == log_buffer.max_length

        # update an iterable object will raise a type error, `log_val` and
        # `count` should be single value
        with pytest.raises(TypeError):
            log_buffer.update(mock_array([1, 2]))

    def test_max_min(self):
        log_history = np.random.randint(1, 5, 20)
        count_history = np.ones(20)
        log_buffer = HistoryBuffer(log_history, count_history)

        window_size = 10
        expected_min = np.min(log_history[-window_size:])
        expected_max = np.max(log_history[-window_size:])
        self.assertEqual(log_buffer.min(window_size), expected_min)
        self.assertEqual(log_buffer.max(window_size), expected_max)

        self.assertEqual(log_buffer.min(), np.min(log_history))
        self.assertEqual(log_buffer.max(), np.max(log_history))

    def test_mean(self):
        log_history = np.random.randint(1, 5, 20)
        count_history = np.ones(20)
        log_buffer = HistoryBuffer(log_history, count_history)

        window_size = 5
        expected_mean = np.sum(log_history[-window_size:]) / np.sum(count_history[-window_size:])
        self.assertAlmostEqual(log_buffer.mean(window_size), expected_mean, places=6)

        expected_mean_full = np.sum(log_history) / np.sum(count_history)
        self.assertAlmostEqual(log_buffer.mean(), expected_mean_full, places=6)

    def test_current(self):
        log_history = np.random.randint(1, 5, 20)
        count_history = np.ones(20)
        log_buffer = HistoryBuffer(log_history, count_history)

        self.assertEqual(log_buffer.current(), log_history[-1])

        empty_buffer = HistoryBuffer()
        with self.assertRaises(ValueError):
            empty_buffer.current()

    def test_statistics(self):
        log_history = np.array([1, 2, 3, 4, 5])
        count_history = np.array([1, 1, 1, 1, 1])
        log_buffer = HistoryBuffer(log_history, count_history)

        self.assertEqual(log_buffer.statistics('mean'), 3)
        self.assertEqual(log_buffer.statistics('min'), 1)
        self.assertEqual(log_buffer.statistics('max'), 5)
        self.assertEqual(log_buffer.statistics('current'), 5)

        with self.assertRaises(KeyError):
            log_buffer.statistics('unknown')

    def test_register_statistics(self):
        log_buffer = HistoryBuffer()
        assert log_buffer.statistics('custom_statistics') == -1