# pylint: disable=unnecessary-lambda-assignment

class AvgMeter(object):
    """Average meter."""

    def __init__(self):
        self._last = 0.
        self._sum = 0.
        self._count = 0

    def reset(self):
        """Reset counter.
        """
        self._last = 0.
        self._sum = 0.
        self._count = 0

    def update(self, value: float, n: int = 1):
        """Update sum and count.

        Args:
            value (float): value.
            n (int): number.
        """
        self._last = value
        self._sum += value * n
        self._count += n

    @property
    def avg(self) -> float:
        """Get average value.

        Returns:
            avg (float)
        """
        return self._sum / self._count if self._count != 0 else 0

    @property
    def last(self) -> float:
        """Get last value.

        Returns:
            last (float)
        """
        return self._last
