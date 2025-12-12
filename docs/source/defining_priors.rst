Defining the prior distribution
===============================

Classes implementing commonly used prior distributions can be found in the
:ref:`midas.priors <priors-ref>` module, however users can create their own custom prior
distributions using the :ref:`BasePrior <BasePrior-ref>` class.

Priors can be applied either to the values of a chosen field, or a particular set of
model parameters.

For example, if we wanted to place a Gaussian prior on the value of the electron
temperature field at specific positions, we could do the following:

.. code-block:: python

    from numpy import array
    from midas import FieldRequest
    from midas.priors import GaussianPrior

    # set up a request for the  value of the electron temperature at each edge of the plasma
    boundary_radius = array([0.35, 1.45])
    boundary_temperature = FieldRequest(name="te", coordinates={"radius": boundary_radius})

    # place a gaussian prior on the requested temperature values
    te_boundary_prior = GaussianPrior(
        name="te_boundary_prior",
        field_positions=boundary_temperature,
        mean=array([0., 0.]),
        standard_deviation=array([10., 5.]),
    )

Or we could place a prior on the value of a calibration parameter:

.. code-block:: python

    from midas import ParameterVector
    calibration_param = ParameterVector(name="calibration_factor", size=1)

    calibration_prior = GaussianPrior(
        name="calibration_prior",
        parameter_vector=calibration_param,
        mean=0.34,
        standard_deviation=0.02
    )