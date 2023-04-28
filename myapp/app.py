from shiny import App, reactive, render, ui
import numpy as np
import pandas as pd
from string import ascii_uppercase
import matplotlib.pyplot as plt


app_ui = ui.page_fluid(
    ui.h2("Barcode Sampling Bias"),
    ui.row(
        ui.column(
            2,
            ui.input_numeric(
                "diversity", "Starting Number of Barcodes:", 1000, min=10, max=10000
            ),
            ui.input_numeric("k", "Carrying Capacity:", 6e6, min=1, max=1e7),
            ui.input_numeric("outgrowth", "Cells Post Sort:", 6e6, min=1, max=1e7),
            ui.input_numeric("mu", "Mean Growth Rate (1/d):", 1, min=0.1, max=100),
            ui.input_numeric(
                "sigma", "Std. Dev. Growth Rate (1/d):", 0.1, min=0.01, max=100
            ),
        ),
        ui.column(
            2,
            ui.input_numeric("passage_num", "Number of Passages:", 5, min=1, max=100),
            ui.input_numeric(
                "days_per_passage", "Days per Passages:", 5, min=1, max=100
            ),
            ui.input_numeric(
                "seeding_num", "Cells Seed per Passage:", 1e6, min=1, max=1e7
            ),
            ui.input_numeric("seed", "Seed:", 9),
            ui.output_text_verbatim("final_lineage_num"),
            ui.output_text_verbatim("time_to_outgrown"),
        ),
        ui.column(
            8,
            ui.output_plot("plot"),
        ),
    ),
)


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
    # get number of days with intrinsic growth rate

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


def simulate(input):
    barcodes = generate_barcodes(start_num=input.diversity())
    growth_rates = generate_growth_rates(
        barcodes, growth_mean=input.mu(), growth_stddev=input.sigma(), seed=input.seed()
    )
    outgrowth, time_to_outgrown = sort_outgrow(growth_rates, input.outgrowth())
    x, y, abundances = grow(
        growth_rates,
        outgrowth,
        days_per_passage=input.days_per_passage(),
        carrying_capacity=input.k(),
        passage_num=input.passage_num(),
        seeding_num=input.seeding_num(),
        seed=input.seed(),
    )

    return x, y, time_to_outgrown, abundances


# %%


def get_max_barcode(abundances):
    return (
        pd.DataFrame(abundances.items(), columns=["barcode", "cells"])
        .sort_values("cells", ascending=False)
        .iloc[0]["barcode"]
    )


def server(input, output, session):
    @reactive.Calc
    def run_simulation():
        return simulate(input)

    @output
    @render.text
    def time_to_outgrown():
        *_, t, _ = run_simulation()
        return t

    @output
    @render.plot
    def plot():
        x, y, *_ = run_simulation()
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel="time (days)", ylabel="N (cells)")
        return fig

    @output
    @render.text
    def final_lineage_num():
        *_, abundances = simulate(input)
        return f"Final # of lineages -> {len(abundances)}\nWinning Lineage: {get_max_barcode(abundances)}"


app = App(app_ui, server)
