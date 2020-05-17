import unittest
import re
import random

import jax
from elicit import State, HyperDistribution, Normal, Uniform, IntervalQuestion, InverseIntervalQuestion, ComparingValuesQuestion

class IntegrationTests(unittest.TestCase):
    def test_im_thinking_of_a_uniform_distribution(self):
        low = 0
        high = 10
        distributions_and_weights = [
            (Uniform(low=low, high=high), 0.5),
            (Normal(loc=(low+high)/2, scale=1), 0.5),
        ]
        questions = [IntervalQuestion(pivot) for pivot in range(low, high)]
        state = State(HyperDistribution(low, high, distributions_and_weights), questions)

        def answer_according_to_uniform(question) -> float:
            pattern = r'How likely is it that the value is < (.*?)\?'
            (number_string,) = re.match(pattern, question.text).groups(1)

            # e.g. the question is "how likely... > 5" and we want to answer "0.5" since we have a
            # uniform distribution over the space.
            return float(number_string) / high

        for _ in range(len(questions)):
            state.step(answer_according_to_uniform)
            'DEBUG'; print(state.hyper_dist.distributions_and_weights)

        [(_, weight_on_uniform), (_, weight_on_normal)] = state.hyper_dist.distributions_and_weights
        self.assertGreater(weight_on_uniform, 0.65)

class QuestionTests(unittest.TestCase):
    def test_interval_question(self):
        question = IntervalQuestion(0) # How likely is the value to be less than zero?
        distribution = Normal(loc=0, scale=1)
        rng_key = jax.random.PRNGKey(0)

        self.assertAlmostEqual(
            question.log_prob(rng_key, distribution, '1'),
            -0.3025,
        )

    def test_inverse_interval_question(self):
        question = InverseIntervalQuestion(0.5) # What's the 50th percentile value?
        distribution = Normal(loc=0, scale=1)
        rng_key = jax.random.PRNGKey(0)

        self.assertAlmostEqual(
            question.log_prob(rng_key, distribution, '1000'),
            -0.25,
        )

        self.assertAlmostEqual(
            question.sample_answer(rng_key, distribution),
            0.06,
            places=2
        )

    def test_comparing_values_question(self):
        question = ComparingValuesQuestion(1, 2) # Value 1 is ___ times more likely than value 2?
        distribution = Uniform(low=0, high=10)
        rng_key = jax.random.PRNGKey(0)

        self.assertAlmostEqual(
            question.log_prob(rng_key, distribution, '1'),
            0,
        )
        self.assertAlmostEqual(
            question.sample_answer(rng_key, distribution),
            1,
        )

if __name__ == "__main__":
    unittest.main()
