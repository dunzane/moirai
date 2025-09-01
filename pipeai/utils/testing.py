
def assert_calls_almost_equal(testcase, actual_calls, expected_calls, places=6):
    """Assert that two lists of mock calls are almost equal, allowing float tolerance."""
    testcase.assertEqual(len(actual_calls), len(expected_calls), "Number of calls does not match.")

    for actual, expected in zip(actual_calls, expected_calls):
        # compare kwargs
        testcase.assertEqual(set(actual.kwargs.keys()), set(expected.kwargs.keys()))
        for k in actual.kwargs:
            v1, v2 = actual.kwargs[k], expected.kwargs[k]
            if isinstance(v1, float) and isinstance(v2, float):
                testcase.assertAlmostEqual(v1, v2, places=places)
            else:
                testcase.assertEqual(v1, v2)

        # compare dicts in args
        testcase.assertEqual(len(actual.args), len(expected.args))
        for d1, d2 in zip(actual.args, expected.args):
            testcase.assertEqual(set(d1.keys()), set(d2.keys()))
            for k in d1:
                v1, v2 = d1[k], d2[k]
                if isinstance(v1, float) and isinstance(v2, float):
                    testcase.assertAlmostEqual(v1, v2, places=places)
                else:
                    testcase.assertEqual(v1, v2)