MIDAS GUI
=========

The MIDAS GUI is a desktop node-graph editor for visually constructing Bayesian
analyses. Nodes represent the data, field models, diagnostic models, likelihoods,
and priors in an analysis. Connecting their typed ports produces a Python script
that can be inspected as the graph changes and exported when the analysis is
ready to run.

Installation and launch
-----------------------

The GUI uses PySide6, which is provided through the optional ``gui`` dependency.
Install MIDAS and the GUI from PyPI with:

.. code-block:: bash

   pip install "midas-fusion[gui]"

For a local checkout of the source repository, install the project with:

.. code-block:: bash

   pip install ".[gui]"

Launch the application using the installed command:

.. code-block:: bash

   midas-gui

Alternatively, invoke the GUI as a Python module:

.. code-block:: bash

   python -m midas_gui

Interface overview
------------------

The main window has four work areas:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Panel
     - Position
     - Purpose
   * - Node Palette
     - Left
     - Browse and search the available node types.
   * - Node Canvas
     - Centre
     - Place nodes and connect them to construct an analysis.
   * - Properties
     - Right
     - Edit the configuration of the selected node.
   * - Code Preview
     - Bottom
     - Inspect and export the Python script generated from the graph.

Panels can be shown or hidden from the :guilabel:`View` menu.

Building an analysis
--------------------

Drag a node from the Node Palette and drop it on the canvas. The palette groups
the built-in nodes by their role, including parameters and data, field models,
diagnostic models, uncertainty models, likelihoods, and priors. Use the search
box above the palette to filter nodes by name.

Input ports appear on the left of a node and output ports on the right. Drag from
an output port to a compatible input port to connect two nodes. Ports are typed,
so the GUI only permits compatible connections, and cycles are not allowed.
Compatible ports are highlighted while a connection is being dragged.

Select a node to edit it in the Properties panel. Editors are chosen from the
property type and include text fields, numeric inputs, ranges, coordinate lists,
and array controls. Array nodes can create values with ``numpy.linspace`` or
``numpy.arange``, hold a constant value, or load data from ``.npy``, ``.npz``,
and ``.csv`` files.

Canvas controls
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Action
     - Effect
   * - Scroll wheel
     - Zoom around the pointer.
   * - Middle-click and drag
     - Pan across the canvas.
   * - Left-click a node
     - Select the node and display its properties.
   * - Delete or Backspace
     - Remove the selected nodes and their connections.
   * - Ctrl+A
     - Select all nodes.
   * - Ctrl+D
     - Duplicate the selected node.
   * - Escape
     - Clear the selection.
   * - Right-click a node
     - Open actions for deleting or disconnecting the node.

Saving and loading sessions
---------------------------

Use :menuselection:`File --> Save` or :menuselection:`File --> Save As...` to
store the graph as a ``.midas`` session file. A session contains the node types,
positions, properties, connections, and paths to imported user modules. Open a
saved session with :menuselection:`File --> Open...`.

Array nodes save their configuration rather than embedding the contents of a
source data file. Files referenced by an array node must therefore remain
available at the saved path when the session is opened again.

Importing custom modules
------------------------

Choose :menuselection:`File --> Import Module...` to load a Python file that
defines custom MIDAS classes. The GUI discovers concrete subclasses of the
supported field model, diagnostic model, likelihood, uncertainty, and prior base
classes and adds them to the Node Palette. Imported module paths are retained in
the session and loaded before its graph is reconstructed.

Code preview and export
-----------------------

The Code Preview updates as nodes, properties, and connections change. Enable
:guilabel:`Runnable template` to include a starting point for optimisation and
sampling in the generated code.

Use :menuselection:`File --> Export Script...` to save the generated analysis as
a ``.py`` file. The exported script imports the required classes and constructs
the graph objects in dependency order.

Settings
--------

Open :menuselection:`File --> Settings...` to adjust the font sizes used by the
Node Palette, Properties panel, and Code Preview, or to select a theme. Theme
changes take effect after restarting the application.