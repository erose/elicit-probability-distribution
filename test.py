import unittest
import re
import random
from elicit import State, HyperDistribution, Normal, Uniform, step

class TestElicit(unittest.TestCase):
    def test_im_thinking_of_a_uniform_distribution(self):
        random.seed(0) # We use randomness to choose which questions to ask.

        distributions_and_weights = [
            (Uniform(low=0, high=10), 0.5),
            (Normal(loc=5, scale=1), 0.5),
        ]
        state = State(HyperDistribution(0, 10, distributions_and_weights))

        def answer_according_to_uniform(question) -> float:
            pattern = r'How likely is it that the value is < (.*?)\?'
            (number_string,) = re.match(pattern, question.text).groups(1)

            # e.g. the question is "how likely... > 5" and we want to answer "0.5" since we have a
            # uniform distribution over the space.
            return float(number_string) / 10.0

        for _ in range(10):
            step(state, answer_according_to_uniform)

        [(_, weight_on_uniform), (_, weight_on_normal)] = state.hyper_dist.distributions_and_weights
        self.assertGreater(weight_on_uniform, 0.8)

if __name__ == "__main__":
    unittest.main()
