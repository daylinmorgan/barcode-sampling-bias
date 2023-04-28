from shiny import App, reactive, render, ui
import matplotlib.pyplot as plt

from simulation import (
    grow,
    generate_barcodes,
    generate_growth_rates,
    sort_outgrow,
    get_max_barcode,
)

app_ui = ui.page_fluid(
    ui.panel_title("Barcode Sampling Bias", "Barcode Sampling Bias"),
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
        return (
            f"Final # of lineages -> {len(abundances)}"
            f"\nWinning Lineage: {get_max_barcode(abundances)}"
        )


app = App(app_ui, server)
