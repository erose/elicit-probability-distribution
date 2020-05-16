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


class Uniform(dist.Uniform):
    def __str__(self):
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
        print(f"Distribution {distribution}, true_frac {true_frac}, reported_frac {reported_frac}")
        return -(true_frac - reported_frac) ** 2

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

        new_weights = jnp.array(self.weights())
        for i, (distribution, weight) in enumerate(self.distributions_and_weights):
            prob = jnp.exp(trust * question.log_prob(rng_key, distribution, answer))
            new_weights = index_update(new_weights, i, weight * prob)
        
        new_weights = new_weights / sum(new_weights)
        return HyperDistribution(self.low, self.high, list(zip(self.distributions(), list(new_weights))))

    def sample(self, rng_key):
        i = dist.Categorical(jnp.array(self.weights())).sample(key=rng_key)
        return self.distributions[i]

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

    def __str__(self):
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

        self.previous_hyper_dist_graphs.append((x, y))
        plt.show()

    def next_question(self):
        unasked_questions = [q for q in self.questions if q not in self.asked_questions]
        return random.choice(unasked_questions)

    def update(self, question, answer):
        # Save the graph so we can show it later.
        self.previous_hyper_dist_graphs.append(self.hyper_dist.graph())

        self.hyper_dist = self.hyper_dist.update(self.next_rng_key(), question, answer)

    def __str__(self):
        return str(self.hyper_dist)


def qa_loop(state):
    while True:
        print(f"Distribution weights: {state}")
        state.plot_current_vs_previous_distributions()
        print()
        question = state.next_question()
        answer = question.ask()
        state.asked_questions.append(question)
        state.update(question, answer)


def normal_and_uniform(low, high):
    return [
        (Uniform(low=low, high=high), 0.5),
        (Normal(loc=(high-low)/2, scale=1), 0.5),
    ]

def normals(low, high, granularity=10.0):
    result = []
    
    step = (high - low) / granularity
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

    distributions_and_weights = normal_and_uniform(low, high)
    hyper_dist = HyperDistribution(low, high, distributions_and_weights)
    state = State(hyper_dist)

    qa_loop(state)
