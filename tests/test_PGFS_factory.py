from probably.analysis.generating_function import GeneratingFunction
from probably.analysis.pgfs import PGFS
import random as rng


def test_bernoulli():
    probability = rng.random()
    assert GeneratingFunction(f"1-{probability} + {probability}* variable") == PGFS.bernoulli("variable", probability)


def test_geometric():
    probability = rng.random()
    assert GeneratingFunction(f"{probability}/(1- (1-{probability})*variable)") == PGFS.geometric("variable",
                                                                                                  probability)


def test_uniform():
    start = rng.randint(0, 10)
    end = start + rng.randint(0, 10)
    assert GeneratingFunction(f"variable**{start} / ({end} - {start}+1) *"
                              f"(variable ** ({end} - {start} + 1) - 1) / (variable - 1)") == \
           PGFS.uniform("variable", start, end)


def test_log():
    probability = rng.random()
    assert GeneratingFunction(f"log(1-{probability}*variable)/log(1-{probability})") == PGFS.log("variable",
                                                                                                 probability)


def test_poisson():
    rate = rng.uniform(0, 10)
    assert GeneratingFunction(f"exp({rate} * (variable -1))") == PGFS.poisson("variable", rate)


def test_binomial():
    n = rng.randint(0, 20)
    p = rng.random()
    assert GeneratingFunction(f"(1-{p}+{p}*variable)**{n}") == PGFS.binomial("variable", n, p)
