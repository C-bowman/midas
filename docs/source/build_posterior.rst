Constructing the posterior
==========================

In order to enable MIDAS to connect seamlessly with other Python tools for optimisation
and uncertainty quantification, we need a function which maps a single 1D array of
parameter values to the posterior log-probability.

To do this, MIDAS inspects the models and priors that are included in the problem
to determine the full set of unique parameters which are required, and creates a
mapping between each parameter and the section of the 1D array it occupies.

To construct the posterior function, we call ``PlasmaState.build_posterior`` and pass
the diagnostic likelihoods, priors and field models we wish to include in the analysis:

.. code-block:: python

    from midas import PlasmaState

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

After calling ``PlasmaState.build_posterior``, we can import the
:ref:`midas.posterior <posterior-ref>`
module, and use its functions to evaluate the posterior or its gradient:

.. code-block:: python

    from midas import posterior

    log_prob = posterior.log_probability(parameter_values)
    log_prob_gradient = posterior.gradient(parameter_values)


Sharing parameters
------------------
Field models, diagnostic models, likelihoods and priors can share parameter vectors.
Declare a ``ParameterVector`` with the same ``name`` and ``size`` in each component's
``Parameters`` collection. MIDAS allocates one slice of the posterior parameter vector
for that name and supplies the same values to every component requesting it. The
``ParameterVector`` objects need not be the same Python object; names identify shared
parameters, and inconsistent sizes are rejected.

For example, two custom field models and a diagnostic model can each request
``Parameters(("calibration", 1))``, alongside their other parameters. The diagnostic
can use ``calibration`` directly while also requesting fields that depend on it.
MIDAS adds the direct derivative and the chain-rule contributions through every
requested field. Priors and parameterised likelihoods follow the same additive rule.

Diagnostic Jacobians and prior gradients must be partial derivatives with respect to
their explicit inputs, holding all other inputs fixed. Do not include a field's
dependence on a shared parameter in the direct parameter derivative: MIDAS accounts
for that dependence separately. Field names must remain unique, and a component's
field names must not collide with its directly requested parameter names.

For code using ``PlasmaState.get_values_and_jacobians`` directly, the third return
value is now a nested dictionary indexed by field name and then parameter name.
The former ``PlasmaState.field_parameter_map`` has been removed because parameters
no longer have a unique owning field.