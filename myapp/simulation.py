import numpy as np
import pandas as pd
from string import ascii_uppercase


def generate_barcodes(start_num):
    num_letters = int(np.ceil(np.log(start_num) / np.log(26)))
    return [
        "".join([ascii_uppercase[int(j)] for j in list(str(i).zfill(num_letters))])
        for i in range(start_num)
    ]


def generate_growth_rates(barcodes, growth_mean, growth_stddev, seed):
    rng = np.random.default_rng(seed)
    growth_rates = rng.normal(loc=growth_mean, scale=growth_stddev, size=len(barcodes))

    return {lineage: g for lineage, g in zip(barcodes, growth_rates)}


def logistic_growth(g, K, N0, t):
    return K / (1 + ((K - N0) / N0) * np.exp(-1 * g * t))


def exponential_growth(g, N0, t):
    return N0 * np.exp(g * t)


def sort_outgrow(growth_rates, total_outgrowth):
    """expand population exponentially"""
    abundances = {barcode: 1 for barcode in growth_rates}
    total = sum(abundances.values())
    # TODO: don't brute force this...
    # back calculate number of days with intrinsic growth rate?

    t = 0
    interval = 1
    while total < total_outgrowth:
        for barcode, g in growth_rates.items():
            abundances[barcode] = exponential_growth(g, 1, t)
        t += interval
        total = sum(abundances.values())

    return abundances, t


def sample(abundances, total, seeding_num, seed):
    """subsample the population to simulate passaging"""

    rng = np.random.default_rng(seed)
    barcodes = list(abundances.keys())
    proportions = list(abundances.values()) / total
    subsampled = {
        barcodes: counts
        for barcodes, counts in zip(
            *np.unique(
                rng.choice(barcodes, int(seeding_num), p=proportions),
                return_counts=True,
            )
        )
    }
    return subsampled


def grow(
    growth_rates,
    outgrowth,
    days_per_passage,
    carrying_capacity,
    passage_num,
    seeding_num,
    seed,
):
    passages = [np.linspace(0, days_per_passage, 100)] * passage_num
    x, y = [], []
    start = sample(outgrowth, sum(outgrowth.values()), seeding_num, seed)

    abundances = start.copy()
    total = sum(abundances.values())

    # must vectorize.... polars?
    for passage, time in enumerate(passages):
        for t in time:
            x.append(t + passage * days_per_passage)
            K = {
                barcode: carrying_capacity * abundance / total
                for barcode, abundance in abundances.items()
            }
            for barcode, g in growth_rates.items():
                if barcode in abundances:
                    abundances[barcode] = logistic_growth(
                        g, K[barcode], start[barcode], t
                    )

            total = sum(abundances.values())
            y.append(total)

        # subsample the population
        abundances = sample(abundances, total, seeding_num, seed)
        start = abundances.copy()

    return x, y, abundances


def get_max_barcode(abundances):
    return (
        pd.DataFrame(abundances.items(), columns=["barcode", "cells"])
        .sort_values("cells", ascending=False)
        .iloc[0]["barcode"]
    )
