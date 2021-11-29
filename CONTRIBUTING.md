<h2> Planned Features </h2>

There are many things to develop, these are the more important ones in order of urgency:

 * Coordinate transformations for performing BO approximations.
 * Coordinate transformations for identifying oscillator modes.
 * Implement hierarchical diagonalisation.
 * Implement Gioele's reduction method for multi-qubit systems.
 * Interface with Huo's AME solver.
 * Sweeping multiple parameters at the same time, not necessarily on a grid.
 * Analysis of the classical potential.
 * Implement time dependence without encroaching on `qutip`'s method of solving time dependence.

These are the less important ones but would increase ease of use significantly:

 * Write documentation on functions that won't change.
 * Improve circuit drawing functionality and useability by separating the drawing from the `CircuitSpec` class.
 * Implement a basic but useful plotting utility based on `matplotlib`.

The original intention was to create a GUI interface for this package. An idea was to use the GTK Webkit interface, which provides internet browser functionality with minimal development effort. With this it is possible to render svg images in different windows and use the GTK widget library to enable control of the underlying data structures. If this can be made to work then it should be straightforward to create a GUI using Glade, however this is very low priority.

<h2> Dev Guide </h2>

To contribute to the source code, just follow the code style and write __reStructuredText__ docstrings as you go along.

To contribute to the documentation generated using __Sphinx__, simply edit the `docs/index.rst` and `docs/conf.py` files, and run

```Shell
make html
```

to build HTML documentation.
