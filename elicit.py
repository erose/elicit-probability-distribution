"""
What distribution am I thinking of? 20 questions for distributions.
"""
from dataclasses import dataclass, field
import math
import time # For debugging.
import random
from typing import *

import jax
import jax.numpy as jnp
from jax.ops import index_update
import numpyro.distributions as dist
import matplotlib.pyplot as plt

from expected_info_gain import eig

class Normal(dist.Normal):
    def __repr__(self):
        return f"Normal({self.loc}, {self.scale})"


class Uniform(dist.Uniform):
    def __repr__(self):
        return f"Uniform({self.low}, {self.high})"


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

    def sample_answer(self, rng_key, distribution: dist.Distribution):
        return distribution.sample(key=rng_key)

    def __eq__(self, other):
        return type(self) == type(other)


class IntervalQuestion(Question):
    pivot: float = 1

    def __init__(self, pivot=1):
        self.pivot = pivot
        super(IntervalQuestion, self).__init__(
            f"How likely is it that the value is < {self.pivot}?"
        )
        
        self._memo = {}

    def calculate_true_frac(self, rng_key, distribution: dist.Distribution):
        num_samples = 100
        samples = distribution.sample(key=rng_key, sample_shape=(num_samples,))
        return jnp.sum(samples < float(self.pivot)) / num_samples

    def true_frac(self, rng_key, distribution: dist.Distribution):
        if distribution not in self._memo:
            # 'DEBUG'; print(f"Miss for {self}, {distribution}.")
            self._memo[distribution] = self.calculate_true_frac(rng_key, distribution)
        return self._memo[distribution]

    def log_prob(self, rng_key, distribution: dist.Distribution, answer: str):
        true_frac = self.true_frac(rng_key, distribution)
        reported_frac = float(answer)

        # For debugging.
        # print(f"Distribution {distribution}, true_frac {true_frac}, reported_frac {reported_frac}")

        return -(true_frac - reported_frac) ** 2

    def sample_answer(self, rng_key, distribution: dist.Distribution):
        return self.true_frac(rng_key, distribution)

    def __eq__(self, other):
        return type(self) == type(other) and self.pivot == other.pivot

@dataclass
class HyperDistribution:
    # Could just be Categorical; this class doesn't need to know that we're dealing with
    # distributions.

    low: float
    high: float
    distributions_and_weights: List[Tuple[dist.Distribution, float]]

    def update(self, rng_key, question: Question, answer: str) -> "HyperDistribution":
        trust = 5 # A scalar that controls how much we update.

        # 'DEBUG'; start = time.time()
        new_weights = jnp.array(self.weights())
        for i, (distribution, weight) in enumerate(self.distributions_and_weights):
            prob = jnp.exp(trust * question.log_prob(rng_key, distribution, answer))
            # 'DEBUG'; print("After prob calculation", time.time() - start)
            new_weights = index_update(new_weights, i, weight * prob)
            # 'DEBUG'; print("After new_weights", time.time() - start)

        # 'DEBUG'; print()
        # 'DEBUG'; print("After loop", time.time() - start)

        new_weights = new_weights / sum(new_weights)
        new_hyper_distribution = HyperDistribution(self.low, self.high, list(zip(self.distributions(), new_weights)))

        # 'DEBUG'; print("After new_hyper_distribution", time.time() - start)
        return new_hyper_distribution

    def sample(self, rng_key):
        i = dist.Categorical(jnp.array(self.weights())).sample(key=rng_key)
        return self.distributions()[i]

    def pdf(self, x: float) -> float:
        weighted_probs = [jnp.exp(d.log_prob(x)) * weight for d, weight in self.distributions_and_weights]
        return sum(weighted_probs)

    def distributions(self):
        return [distribution for distribution, weight in self.distributions_and_weights]

    def weights(self):
        return [weight for distribution, weight in self.distributions_and_weights]

    def graph(self) -> Tuple[jnp.array]:
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

    def __repr__(self):
        return str(self.weights())


class State:
    hyper_dist: HyperDistribution
    previous_hyper_dist_graphs: List[Tuple]
    questions: List[Question]
    asked_questions: List[Question]
    rng_key: jax.random.PRNGKey

    def __init__(self, hyper_dist):
        self.hyper_dist = hyper_dist
        self.previous_hyper_dist_graphs = []
        self.rng_key = jax.random.PRNGKey(0)
        self.questions = [
            IntervalQuestion(pivot) for pivot in range(hyper_dist.low, hyper_dist.high)
        ]  # + [SampleQuestion()]
        self.asked_questions = []

    def next_rng_key(self):
        current_key, next_rng_key = jax.random.split(self.rng_key, 2)
        self.rng_key = next_rng_key
        return current_key

    def plot_current_vs_previous_distributions(self):
        x, y = self.hyper_dist.graph()
        plt.plot(x, y)

        # Plot previous distributions with a progressively lighter opacity, as defined by a starting
        # point and a delta.
        base_alpha = 0.15
        alpha_delta = 5
        for i, (x, y) in enumerate(reversed(self.previous_hyper_dist_graphs)):
            alpha = base_alpha - alpha_delta * (0.01 * i)
            if alpha <= 0:
                continue # No point in plotting this; it wouldn't be visible.

            plt.plot(x, y, color='red', alpha=alpha)

        plt.show()

    def next_question(self) -> Optional[Question]:
        """
        Return the question with highest expected information gain.
        """
        unasked_questions = [q for q in self.questions if q not in self.asked_questions]
        if len(unasked_questions) == 0:
            return None

        max_eig_question = max(unasked_questions, key=lambda q: eig(self.next_rng_key(), self.hyper_dist, q))
        return max_eig_question

    def update(self, question, answer):
        self.hyper_dist = self.hyper_dist.update(self.next_rng_key(), question, answer)

    def save_graph(self):
        self.previous_hyper_dist_graphs.append(self.hyper_dist.graph())

    def __repr__(self):
        return str(self.hyper_dist)

def step(state, get_answer: Callable[[Question], float]) -> None:
    'DEBUG'; start = time.time()
    question = state.next_question()
    if question is None:
        print("No more questions to ask!")
        return
    'DEBUG'; print("next_question took", time.time() - start)
    
    answer = get_answer(question)
    state.asked_questions.append(question)
    state.save_graph() # For visual comparison.
    state.update(question, answer)

class Priors:
    @staticmethod
    def uniform_vs_normal(low, high):
        return [
            (Uniform(low=low, high=high), 0.5),
            (Normal(loc=(high-low)/2, scale=1), 0.5),
        ]

    @staticmethod
    def normals(low, high, granularity=10.0):
        """
        Normals with a constant scale.
        """

        result = []
        
        step = (high - low) / granularity
        n = granularity + 1
        scale = step
        
        loc = low
        while loc <= high:
            result.append(
                (Normal(loc=loc, scale=scale), 1 / n)
            )
            loc += step

        return result

def qa_loop(state):
    while True:
        print(f"Distribution weights: {state}")
        state.plot_current_vs_previous_distributions()
        print()
        
        step(state, get_answer=lambda question: question.ask())

if __name__ == "__main__":
    low = 0#int(input("What is the lowest value you want to consider? (Round to an integer.)\n"))
    high = 10#int(input("What is the highest value you want to consider? (Round to an integer.)\n"))

    hyper_dist = HyperDistribution(low, high, Priors.uniform_vs_normal(low, high))
    state = State(hyper_dist)

    qa_loop(state)
