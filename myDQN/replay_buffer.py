from _collections import deque
import random


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()

    def save_transition(self, transition):
        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)


if __name__ == '__main__':
    buffer = ReplayBuffer(100)
