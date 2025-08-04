import time


class TimePredictor:
    """TimePredictor"""

    def __init__(self, start_step: int, end_step: int):
        self.start_step = start_step
        self.end_step = end_step
        self.start_time = time.time()

    def get_remaining_time(self, step: int) -> float:
        now_time = time.time()
        return (now_time - self.start_time) * (self.end_step - self.start_step) / (step - self.start_step)

    def get_expected_end_time(self, step: int) -> float:
        return self.start_time + self.get_remaining_time(step)
