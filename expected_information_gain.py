import math
import random
import time

import jax
from tqdm import tqdm

def kl_divergence(P: "HyperDistribution", Q: "HyperDistribution"):
    # FIXME: Handle q = 0
    # FIXME: Should apply to arbitrary discrete distributions
    kl = 0.0
    for (p, q) in zip(P.weights(), Q.weights()):
        if p == 0:
            continue
        kl += p * math.log(p / q)
    return kl

def expected_information_gain(rng_key, hyper_dist, question, previous_answers) -> float:
    """
    Expected information gain of asking the given question in the current state.
    """
    num_samples = 10
    info_gains = []
    for _ in range(num_samples):
        dist_key, answer_key, update_key, rng_key = jax.random.split(rng_key, 4)
        
        # Suppose the true distribution is this
        dist = hyper_dist.sample(dist_key)

        # Then what answer do I expect? If this question has already been answered, have a pretty
        # high prior that the answer will be the same as the last time it was asked. And if it's
        # been answered twice, assume the answer will be the same as the last time it was asked.
        human_responds_same_prior = 0.9
        if len(previous_answers) > 1:
            answer = previous_answers[-1]
        if len(previous_answers) > 0 and random.random() < human_responds_same_prior:
            answer = previous_answers[-1]
        else:
            answer = question.sample_answer(answer_key, dist)

        # Given that answer, what's my new posterior on distributions?
        hyper_posterior = hyper_dist.update(update_key, question, answer)

        # How much information have I gained from asking this question?
        info_gain = kl_divergence(hyper_posterior, hyper_dist)
        info_gains.append(info_gain)
    
    return sum(info_gains) / len(info_gains)
