# MIDAS GUI

The MIDAS GUI is a desktop node-graph editor for visually constructing Bayesian analyses using the `midas` framework. Rather than writing Python scripts from scratch, users place
nodes representing diagnostic models, field models, priors and likelihoods on a
canvas, wire them together, and export a ready-to-run Python script.

## Key Features

- **Visual node-graph editor** — drag-and-drop nodes onto an interactive canvas
  and connect them by wiring output ports to input ports.
- **Custom module import** — import your own Python modules containing MIDAS
  subclasses (e.g. custom diagnostic models or priors) and they appear
  automatically in the node palette.
- **Live code preview** — a Python script is generated in real time as the graph
  is built, with optional explanatory comments and runnable template stubs.
- **Script export** — export the generated analysis script as a `.py` file.
- **Session save/load** — persist and restore your full graph layout, wiring and
  properties as `.midas` session files (JSON format).
- **Custom module import** — import your own Python modules containing MIDAS
  subclasses (e.g. custom diagnostic models or priors) and they appear
  automatically in the node palette.

## Launching the GUI

The GUI is not yet published to PyPI, so you need to install from the source
repository. Clone the `gui-experiment` branch and install with the `gui` extra:

```
git clone -b gui-experiment https://github.com/C-bowman/midas.git
cd midas
pip install ".[gui]"
```

There are two ways to launch the application:

**1. Entry-point command (recommended)**

```
midas-gui
```

If your terminal reports that `midas-gui` is not recognised, your Python
`Scripts` directory is not on your system `PATH`. You can either add it
(pip will print the relevant directory in its install output) or use the
fallback below.

**2. Module invocation (always works)**

```
python -m midas_gui
```

## Using the GUI

### Window Layout

The main window is divided into four panels:

| Panel | Position | Purpose |
|---|---|---|
| **Node Palette** | Left dock | Browse and search available node types |
| **Node Canvas** | Centre | Main workspace — place nodes and draw wires |
| **Properties Panel** | Right dock | Edit the properties of the selected node |
| **Code Preview** | Bottom dock | View the live-generated Python script |

Each dock panel can be toggled on or off from the **View** menu.

### Adding Nodes

The **Node Palette** on the left lists every available node type, grouped into
categories:

- **Parameters & Data** — `ParameterVector`, `Array`, `Coordinates`,
  `FieldRequest`
- **Field Models** — `PiecewiseLinearField`, `CubicSplineField`,
  `BSplineField`, `ExSplineField`, `TriangularMeshField`
- **Diagnostic Models** — `LinearDiagnosticModel`
- **Uncertainty Models** — `ConstantUncertainty`, `LinearUncertainty`
- **Likelihoods** — `DiagnosticLikelihood`, `GaussianLikelihood`,
  `LogisticLikelihood`, `CauchyLikelihood`
- **Priors** — `GaussianPrior`, `ExponentialPrior`, `BetaPrior`,
  `GaussianProcessPrior`, `SoftLimitPrior`

To add a node, **drag** it from the palette and **drop** it onto the canvas. A
search bar at the top of the palette lets you filter nodes by name.

### Wiring Nodes Together

Each node can have **input ports** (left side) and **output ports** (right
side). Click and drag from an output port to draw a temporary wire, then release
on a compatible input port to create a connection.

During a drag, compatible ports glow green and incompatible ports dim out. Ports
are typed — for example a `Field` output can only connect to a `Field` input —
and the canvas enforces acyclic connections.

### Editing Properties

Click a node on the canvas to select it. Its editable properties appear in the
**Properties Panel** on the right. The available editors depend on the node type:

- **Text fields** for string properties (e.g. field names, axis names).
- **Spin boxes** for numeric properties (integers and floats).
- **Range editors** for tuple properties (e.g. prior bounds).
- **Array editors** for `Array` nodes, offering three data-source modes:
  - *linspace* — specify start, stop and number of points.
  - *arange* — specify start, stop and step size.
  - *file* — browse for a `.npy`, `.npz` or `.csv` file.
- **Coordinate list** for `Coordinates` nodes — add or remove named coordinates
  with dedicated buttons.

For auto-generated nodes, only configuration properties (those that are not wired
inputs) are shown. If a parameter has an unrecognised type, a text field for a
variable name and an optional "Use default" checkbox are provided.

### Canvas Navigation

| Action | Effect |
|---|---|
| **Scroll wheel** | Zoom in / out (anchored at the mouse position) |
| **Middle-click + drag** | Pan the viewport |
| **Left-click a node** | Select it (updates Properties and Code Preview) |
| **Delete key** | Remove selected nodes and their wires |
| **Escape** | Deselect all |
| **Right-click a node** | Context menu (delete, duplicate, disconnect all) |
| **Right-click the canvas** | Context menu (add node, paste, select all) |

### Code Preview

The **Code Preview** panel at the bottom displays the generated Python script,
updated live as you modify the graph. It includes:

- A **"Runnable template"** checkbox — when enabled, the script includes
  commented stubs for running optimisation and MCMC sampling.
- An **"Include comments"** checkbox — toggles explanatory inline comments.
- An **"Export .py"** button — saves the script to a file.

The preview has syntax highlighting (keywords, strings and comments) and uses
the Cascadia Code or Consolas font.

### Saving and Loading Sessions

Use the **File** menu (or keyboard shortcuts) to manage sessions:

| Action | Shortcut |
|---|---|
| **Open…** | Ctrl+O |
| **Save** | Ctrl+S |
| **Save As…** | — |

Sessions are stored as `.midas` files (JSON). A session captures the full graph
(node types, positions, property values, wiring) together with the paths of any
imported user modules. When a session is loaded, imported modules are
re-imported automatically so that custom node types are available before the
graph is reconstructed.

Note that `Array` nodes store their *configuration* (e.g. linspace parameters or
file paths), not raw numpy data. If an array was loaded from a file, that file
must still be accessible at the stored path when the session is re-opened.

### Importing Custom Modules

Select **File → Import Module…** and choose a `.py` file containing classes that
inherit from one of the supported MIDAS base classes (`DiagnosticModel`,
`FieldModel`, `LikelihoodFunction`, `UncertaintyModel` or `BasePrior`). The GUI
inspects the module, registers any valid subclasses as new node types, and
refreshes the palette. Imported module paths are saved with the session so they
are automatically reloaded next time.

### Exporting a Script

Click the **"Export .py"** button in the Code Preview panel (or use
**File → Export Script…**) to save the generated Python script. The exported
script is a standalone file that imports MIDAS, constructs every object in
dependency order, and calls `PlasmaState.build_posterior()`.

### Settings

Open **File → Settings…** to adjust:

- **Font sizes** — independent font-size selection for the Node Palette,
  Properties Panel and Code Preview.
- **Theme** — select a colour theme (default is the "Deep Ocean" dark theme). Theme changes take effect on restart.
