"""
What distribution am I thinking of? 20 questions for distributions.
"""
from dataclasses import dataclass, field
import math
import random
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax.ops import index_update
import numpyro.distributions as dist
import matplotlib.pyplot as plt

class Normal(dist.Normal):
    def __str__(self):
        return f"Normal({self.loc}, {self.scale})"


@dataclass
class Question:
    text: str

    def ask(self):
        return input(self.text + " ")

    def log_prob(self, rng_key, distribution: dist.Distribution, answer: str):
        """
        Log likelihood of observing this answer for this distribution
        """
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError


class SampleQuestion(Question):
    def __init__(self):
        super(SampleQuestion, self).__init__("What might a typical value look like?")

    def log_prob(self, rng_key, distribution: dist.Distribution, answer: str):
        return distribution.log_prob(float(answer))

    def __eq__(self, other):
        return type(self) == type(other)


class IntervalQuestion(Question):
    pivot: float = 1

    def __init__(self, pivot=1):
        self.pivot = pivot
        super(IntervalQuestion, self).__init__(
            f"How likely is it that the value is < {self.pivot}?"
        )

    def true_frac(self, rng_key, distribution: dist.Distribution):
        num_samples = 100
        samples = distribution.sample(key=rng_key, sample_shape=(num_samples,))
        return jnp.sum(samples < float(self.pivot)) / num_samples

    def log_prob(self, rng_key, distribution: dist.Distribution, answer: str):
        true_frac = self.true_frac(rng_key, distribution)
        reported_frac = float(answer)
        return -10 * ((true_frac - reported_frac) ** 2) # TODO: I added -10 based on no principled reason.

    def __eq__(self, other):
        return type(self) == type(other) and self.pivot == other.pivot

@dataclass
class HyperDistribution:
    # Could just be Categorical; this class doesn't need to know that we're dealing with
    # distributions.

    low: float
    high: float
    distributions: List[dist.Distribution]
    weights: List[float]

    def update(self, rng_key, question: Question, answer: str) -> "HyperDistribution":
        new_weights = jnp.array(self.weights)
        for i, (distribution, weight) in enumerate(
            zip(self.distributions, self.weights)
        ):
            prob = jnp.exp(question.log_prob(rng_key, distribution, answer))
            new_weights = index_update(new_weights, i, weight * prob)
        new_weights = new_weights / sum(new_weights)
        return HyperDistribution(self.low, self.high, self.distributions, new_weights)

    def sample(self, rng_key):
        i = dist.Categorical(jnp.array(self.weights)).sample(key=rng_key)
        return self.distributions[i]

    def pdf(self, x: float) -> float:
        distributions_and_weights = zip(self.distributions, self.weights)
        weighted_probs = [jnp.exp(d.log_prob(x)) * weight for d, weight in distributions_and_weights]
        return sum(weighted_probs)

    def pdf_pairs(self) -> Tuple[jnp.array]:
        xs = []
        ys = []
        step = (self.high - self.low) / 100.0
        
        x = self.low
        while x < self.high:
            xs.append(x)
            y = self.pdf(x)
            ys.append(y)

            x += step
        
        return (xs, ys)

    def __str__(self):
        return str(self.weights)


class State:
    hyper_dist: HyperDistribution
    questions: List[Question]
    asked_questions: List[Question]
    rng_key: jax.random.PRNGKey

    def __init__(self, hyper_dist):
        self.hyper_dist = hyper_dist
        self.rng_key = jax.random.PRNGKey(0)
        self.questions = [
            IntervalQuestion(pivot) for pivot in range(hyper_dist.low, hyper_dist.high)
        ]  # + [SampleQuestion()]
        self.asked_questions = []

    def next_rng_key(self):
        current_key, next_rng_key = jax.random.split(self.rng_key, 2)
        self.rng_key = next_rng_key
        return current_key

    def plot_current_distribution(self):
        x, y = self.hyper_dist.pdf_pairs()
        plt.plot(x, y)
        plt.show()

    def next_question(self):
        unasked_questions = [q for q in self.questions if q not in self.asked_questions]
        return random.choice(unasked_questions)

    def update(self, question, answer):
        self.hyper_dist = self.hyper_dist.update(self.next_rng_key(), question, answer)

    def __str__(self):
        return str(self.hyper_dist)


def qa_loop(state):
    while True:
        print(f"Distribution weights: {state}")
        state.plot_current_distribution()
        print()
        question = state.next_question()
        answer = question.ask()
        state.asked_questions.append(question)
        state.update(question, answer)


def initial_distributions(low, high):
    result = []
    step = (high - low) / 10.0

    # Add normals.
    loc = low
    while loc <= high:
        # TODO: Add different scales. How high to push scale?
        for scale in (1,):
            result.append(Normal(loc=loc, scale=scale))
        loc += step

    return result

if __name__ == "__main__":
    low = 0#int(input("What is the lowest value (integer) you want to consider? "))
    high = 10#int(input("What is the highest value (integer) you want to consider? "))

    distributions = initial_distributions(low, high)
    n = len(distributions)
    hyper_dist = HyperDistribution(low, high, distributions, [1 / n] * n)
    state = State(hyper_dist)

    qa_loop(state)
