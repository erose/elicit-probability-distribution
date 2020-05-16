import math
import time

import jax
from tqdm import tqdm

def kl_divergence(P: "HyperDistribution", Q: "HyperDistribution"):
    # FIXME: Handle q = 0
    # FIXME: Should apply to arbitrary discrete distributions
    kl = 0.0
    for (p, q) in zip(P.weights(), Q.weights()):
        kl += p * math.log(p / q)
    return kl

def eig(rng_key, hyper_dist, question) -> float:
    """
    Expected information gain of asking the given question in the current state.
    """
    num_samples = 100
    info_gains = []
    for _ in range(num_samples):
        start = time.time()

        dist_key, answer_key, update_key, rng_key = jax.random.split(rng_key, 4)
        # Suppose the true distribution is this
        dist = hyper_dist.sample(dist_key)
        print("After hyper_dist.sample: ", time.time() - start)

        # Then what answer do I expect?
        answer = question.sample_answer(answer_key, dist)
        print("After question.sample_answer: ", time.time() - start)

        # Given that answer, what's my new posterior on distributions?
        hyper_posterior = hyper_dist.update(update_key, question, answer)
        print("After hyper_dist.update: ", time.time() - start)

        # How much information have I gained from asking this question?
        info_gain = kl_divergence(hyper_posterior, hyper_dist)
        info_gains.append(info_gain)

        print("Overall: ", time.time() - start)
    
    return sum(info_gains) / len(info_gains)
