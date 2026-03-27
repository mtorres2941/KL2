# Exploration of EPD distribution fitting using MCMC

## Setup

I used [uv](https://docs.astral.sh/uv/) to set up the Python environment.  It should be as simple as running this command to start a jupyterlab server and set up the environment with the necessary packages installed:

```
uvx --with pyproject-local-kernel --with ipysankeywidget --from jupyterlab jupyter-lab
```

## Notebooks

`Basic pymc test.ipynb` and `Basic DPGMM.ipynb` are basically from the [PyMC documentation](https://www.pymc.io/projects/examples/en/latest/mixture_models/dp_mix.html) (with some additional plots/exploration added)

`Mock EPD data experiment.ipynb` uses mock EPD GWP data to fit different versions of the model:

1. Original DPGMM mixture model
2. Original DPGMM mixture model with additional industry-average observation (doesn't work)
3. Extended DPGMM mixture model with additional "representative correction factor" parameter and industry-average observation