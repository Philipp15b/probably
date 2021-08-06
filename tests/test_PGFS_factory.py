from probably.analysis.generating_function import GeneratingFunction
from probably.analysis.pgfs import PGFS
import random as rng
import pytest


def test_bernoulli():
    probability = str(rng.random())
    assert GeneratingFunction(f"1-{probability} + {probability}* variable") == PGFS.bernoulli("variable", probability)


def test_geometric():
    probability = str(rng.random())
    assert GeneratingFunction(f"{probability}/(1- (1-{probability})*variable)") == PGFS.geometric("variable",
                                                                                                  probability)


def test_uniform():
    start = rng.randint(0, 10)
    end = start + rng.randint(0, 10)
    assert GeneratingFunction(f"variable**{start} / ({end} - {start}+1) *"
                              f"(variable ** ({end} - {start} + 1) - 1) / (variable - 1)") == \
           PGFS.uniform("variable", str(start), str(end))


def test_log():
    probability = str(rng.random())
    assert GeneratingFunction(f"log(1-{probability}*variable)/log(1-{probability})") == PGFS.log("variable",
                                                                                                 probability)


def test_poisson():
    rate = str(rng.randint(0, 10))
    hand = GeneratingFunction(f"exp({rate} * (variable -1))")
    factory = PGFS.poisson("variable", str(rate))
    equal = hand == factory
    assert equal


def test_binomial():
    n = rng.randint(0, 20)
    p = rng.random()
    assert GeneratingFunction(f"(1-{p}+{p}*variable)**{n}") == PGFS.binomial("variable", str(n), str(p))
