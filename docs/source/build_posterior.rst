Constructing the posterior
==========================

In order to enable MIDAS to connect seamlessly with other Python tools for optimisation
and uncertainty quantification, we need a function which maps a single 1D array of
parameter values to the posterior log-probability.

To do this, MIDAS inspects the models and priors that are included in the problem
to determine the full set of unique parameters which are required, and creates a
mapping between each parameter and the section of the 1D array it occupies.

To construct the posterior function, we call `PlasmaState.build_posterior` and pass
any diagnostic models, priors and field models we wish to include in the analysis:

.. code-block:: python

    # collect all the diagnostics we want to include in the analysis
    diagnostics = [brem_likelihood, pressure_likelihood, interferometer_likelihood]

    # collect all the priors we want to include in the analysis
    priors = [te_gp, ne_gp, te_boundary_prior, ne_boundary_prior]

    # collect models for the fields that are requested by the diagnostics
    field_models = [te_field_model, ne_field_model]

    # Use the collected models and priors to build the posterior distribution
    PlasmaState.build_posterior(
        diagnostics=diagnostics,
        priors=priors,
        field_models=field_models,
    )

After calling `PlasmaState.build_posterior`, we can import the :ref:`midas.posterior <posterior-ref>`
module, and use its functions to evaluate the posterior or its gradient:

.. code-block:: python

    from midas import posterior

    log_prob = posterior.log_probability(parameter_values)
    log_prob_gradient = posterior.gradient(parameter_values)