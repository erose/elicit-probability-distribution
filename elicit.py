"""
What distribution am I thinking of? 20 questions for distributions.
"""
from dataclasses import dataclass, field
import math
import time # For debugging.
import random
import collections
from typing import *

import jax
import jax.numpy as jnp
from jax.ops import index_update
import numpyro.distributions as dist
import matplotlib.pyplot as plt
from tqdm import tqdm

from expected_information_gain import expected_information_gain

class Normal(dist.Normal):
    def __repr__(self):
        return f"Normal({self.loc}, {self.scale})"


class Uniform(dist.Uniform):
    def __repr__(self):
        return f"Uniform({self.low}, {self.high})"


class Exponential(dist.Exponential):
    def __repr__(self):
        return f"Exponential({self.rate})"


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

    def __hash__(self):
        return hash(str(self))


class SampleQuestion(Question):
    def __init__(self):
        super().__init__("What might a typical value look like?")

    def log_prob(self, rng_key, distribution: dist.Distribution, answer: str):
        return distribution.log_prob(float(answer))

    def sample_answer(self, rng_key, distribution: dist.Distribution):
        return distribution.sample(key=rng_key)


class IntervalQuestion(Question):
    pivot: float

    def __init__(self, pivot):
        self.pivot = pivot
        self._memo = {} # TODO: Explain.
        super().__init__(f"How likely is it that the value is < {self.pivot}?")

    def log_prob(self, rng_key, distribution: dist.Distribution, answer: str):
        true_frac = self.true_frac(rng_key, distribution)
        reported_frac = float(answer)

        # 'DEBUG'; print(f"Distribution {distribution}, true_frac {true_frac}, reported_frac {reported_frac}")
        return -(true_frac - reported_frac) ** 2

    def sample_answer(self, rng_key, distribution: dist.Distribution):
        return self.true_frac(rng_key, distribution)

    def calculate_true_frac(self, rng_key, distribution: dist.Distribution):
        num_samples = 100
        samples = distribution.sample(key=rng_key, sample_shape=(num_samples,))
        return jnp.sum(samples < float(self.pivot)) / num_samples

    def true_frac(self, rng_key, distribution: dist.Distribution):
        if distribution not in self._memo:
            # 'DEBUG'; print(f"Miss for {self}, {distribution}.")
            self._memo[distribution] = self.calculate_true_frac(rng_key, distribution)
        return self._memo[distribution]


class InverseIntervalQuestion(Question):
    probability: float

    def __init__(self, probability):
        self.probability = probability
        super().__init__(
            f"What's the {self.percentile()}th percentile value? i.e. what is X such that there is a {self.percentile()}% chance that the value is less than X?"
        )

    def log_prob(self, rng_key, distribution: dist.Distribution, answer: str):
        num_samples = 100
        samples = distribution.sample(key=rng_key, sample_shape=(num_samples,))

        true_frac = jnp.sum(samples < float(answer)) / num_samples
        # 'DEBUG'; print("true_frac", true_frac)
        supposed_frac = self.probability
        # 'DEBUG'; print("supposed_frac", supposed_frac)

        # TODO: Not sure why I do the squaring thing here, just copying IntervalQuestion.
        return -(true_frac - supposed_frac) ** 2

    def sample_answer(self, rng_key, distribution: dist.Distribution):
        num_samples = 100
        samples = distribution.sample(key=rng_key, sample_shape=(num_samples,))

        return jnp.percentile(samples, self.percentile())

    def percentile(self):
        return int(self.probability * 100)


class ComparingValuesQuestion(Question):
    value_1: float
    value_2: float

    def __init__(self, value_1, value_2):
        self.value_1 = value_1
        self.value_2 = value_2
        super().__init__(
            f"Value {self.value_1} is ___ times more likely than value {self.value_2}?"
        )

    def log_prob(self, rng_key, distribution: dist.Distribution, answer: str):
        # TODO: Not sure why I do the squaring thing here, just copying IntervalQuestion.
        return -(self.true_ratio(distribution) - float(answer)) ** 2

    def sample_answer(self, rng_key, distribution: dist.Distribution):
        return self.true_ratio(distribution)

    def true_ratio(self, distribution):
        value_1_prob = jnp.exp(distribution.log_prob(self.value_1))
        value_2_prob = jnp.exp(distribution.log_prob(self.value_2))
        
        return value_1_prob / value_2_prob


@dataclass
class HyperDistribution:
    # Could just be Categorical; this class doesn't need to know that we're dealing with
    # distributions.

    low: float
    high: float
    distributions_and_weights: List[Tuple[dist.Distribution, float]]

    def __init__(self, low, high, distributions_and_weights):
        self.low = low
        self.high = high
        self.distributions_and_weights = distributions_and_weights

    def update(self, rng_key, question: Question, answer: str) -> "HyperDistribution":
        trust = 1 # A scalar that controls how much we update.

        # 'DEBUG'; start = time.time()
        new_weights = jnp.array(self.weights())
        for i, (distribution, weight) in enumerate(self.distributions_and_weights):
            # Optimization: if a weight is low enough, it isn't going to be relevant again so just
            # set it to zero and move on.
            if weight < 1e-4:
                new_weights = index_update(new_weights, i, 0)
                continue

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

    def mean(self):
        return sum(d.mean * weight for d, weight in self.distributions_and_weights)

    def distributions(self):
        return [distribution for distribution, weight in self.distributions_and_weights]

    def weights(self):
        return [weight for distribution, weight in self.distributions_and_weights]

    def most_likely_distribution_and_weight(self):
        return max(self.distributions_and_weights, key=lambda t: t[1])

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
    answers_to_questions: Dict[Question, List[float]]
    rng_key: jax.random.PRNGKey
    num_steps: int

    def __init__(self, hyper_dist, questions):
        self.hyper_dist = hyper_dist
        self.previous_hyper_dist_graphs = []
        self.rng_key = jax.random.PRNGKey(0)
        self.questions = questions
        self.answers_to_questions = collections.defaultdict(list)
        self.num_steps = 0

    def step(self, get_answer: Callable[[Question], float]) -> None:
        'DEBUG'; start = time.time()
        question = self.next_question()
        'DEBUG'; print("next_question took", time.time() - start)
        
        if question is None:
            print("No more questions to ask!")
            return
        else:
            answer = get_answer(question)
        
        self.save_graph() # For visual comparison.
        self.update(question, answer)

        self.num_steps += 1

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
        # We'll determine the next question to ask by using a combination of heuristics and expected
        # information gain calculations. First, we'll assemble a list of candidate questions, drawn
        # from the pool of all questions, using heuristics.
        candidates = []

        # TODO: testing.
        interval_questions = [q for q in self.questions if isinstance(q, IntervalQuestion)]
        candidates.extend(interval_questions)

        # TODO: testing.
        comparing_values_questions = [q for q in self.questions if isinstance(q, ComparingValuesQuestion)]
        candidates.extend(comparing_values_questions)

        # TODO: testing
        inverse_interval_questions = [q for q in self.questions if isinstance(q, InverseIntervalQuestion)]
        candidates.extend(inverse_interval_questions)

        # TODO: testing.
        sample_questions = [q for q in self.questions if isinstance(q, SampleQuestion)]
        candidates.extend(sample_questions)

        # Filter out questions we've already asked more than once.
        candidates = [q for q in candidates if len(self.answers_to_questions[q]) < 2]

        # Of the candidates, return the question with highest expected information gain.
        candidates_to_eigs = {}
        for question in tqdm(candidates):
            previous_answers = self.answers_to_questions[question]
            eig = expected_information_gain(self.next_rng_key(), self.hyper_dist, question, previous_answers)
            candidates_to_eigs[question] = eig
            # 'DEBUG'; print(question, eig)

        maximum_eig_question = max(candidates, key=lambda q: candidates_to_eigs[q])
        # 'DEBUG'; print("Chosen is", maximum_eig_question)
        # 'DEBUG'; print("Expected information gain is", candidates_to_eigs[maximum_eig_question])
        return maximum_eig_question

    def update(self, question, answer):
        self.hyper_dist = self.hyper_dist.update(self.next_rng_key(), question, answer)

        # Save the answer to the question.
        self.answers_to_questions[question].append(answer)

    def save_graph(self):
        self.previous_hyper_dist_graphs.append(self.hyper_dist.graph())

    def __repr__(self):
        return str(self.hyper_dist)

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
            result.append((Normal(loc=loc, scale=scale), 1 / n))
            loc += step

        return result

    @staticmethod
    def everything(low, high):
        delta = high - low
        scale_1_normals = [
            (Normal(loc=low + 0*delta, scale=1), 1/33),
            (Normal(loc=low + (1/10)*delta, scale=1), 1/33),
            (Normal(loc=low + (2/10)*delta, scale=1), 1/33),
            (Normal(loc=low + (3/10)*delta, scale=1), 1/33),
            (Normal(loc=low + (4/10)*delta, scale=1), 1/33),
            (Normal(loc=low + (5/10)*delta, scale=1), 1/33),
            (Normal(loc=low + (6/10)*delta, scale=1), 1/33),
            (Normal(loc=low + (7/10)*delta, scale=1), 1/33),
            (Normal(loc=low + (8/10)*delta, scale=1), 1/33),
            (Normal(loc=low + (9/10)*delta, scale=1), 1/33),
            (Normal(loc=low + (10/10)*delta, scale=1), 1/33),
        ]
        scale_2_normals = [
            (Normal(loc=low + 0*delta, scale=2), 1/33),
            (Normal(loc=low + (1/10)*delta, scale=2), 1/33),
            (Normal(loc=low + (2/10)*delta, scale=2), 1/33),
            (Normal(loc=low + (3/10)*delta, scale=2), 1/33),
            (Normal(loc=low + (4/10)*delta, scale=2), 1/33),
            (Normal(loc=low + (5/10)*delta, scale=2), 1/33),
            (Normal(loc=low + (6/10)*delta, scale=2), 1/33),
            (Normal(loc=low + (7/10)*delta, scale=2), 1/33),
            (Normal(loc=low + (8/10)*delta, scale=2), 1/33),
            (Normal(loc=low + (9/10)*delta, scale=2), 1/33),
            (Normal(loc=low + (10/10)*delta, scale=2), 1/33),
        ]

        return [
            (Uniform(low=low, high=high), 1/3),
            *scale_1_normals,
            *scale_2_normals,
        ]

def qa_loop(state):
    while True:
        print()
        print("Here's my current state of mind:")
        # Print the distributions in descending order of likelihood.
        for d, w in sorted(state.hyper_dist.distributions_and_weights, key=lambda t: -t[1]):
            # Skip 'dead' distributions.
            if w == 0:
                continue
            print("Distribution:", d, "Weight:", w)
        state.plot_current_vs_previous_distributions()
        print()

        most_likely_distribution, weight = state.hyper_dist.most_likely_distribution_and_weight()
        threshold = 0.95
        if weight > threshold:
            print(f"With probability >= {threshold}, I guess you're thinking of: ", most_likely_distribution)
            break
        
        state.step(get_answer=lambda question: question.ask())

if __name__ == "__main__":
    low = 0#int(input("What is the lowest value you want to consider? (Round to an integer.)\n"))
    high = 10#int(input("What is the highest value you want to consider? (Round to an integer.)\n"))

    distributions_and_weights = Priors.everything(low, high)
    'DEBUG'; print("Weights add to", sum(w for d, w in distributions_and_weights))
    questions = [
        *[IntervalQuestion(pivot) for pivot in range(low, high)],
        SampleQuestion(),
        *[InverseIntervalQuestion(percentile * 0.01) for percentile in range(0, 100 + 1, 10)],
        # *[ComparingValuesQuestion(value_1, value_2) for value_1, value_2 in zip(range(low, high), range(low+1, high+1))]
    ]
    state = State(HyperDistribution(low, high, Priors.everything(low, high)), questions)

    qa_loop(state)
