from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import BoundaryNorm, Normalize
import numpy as np


class ColorMapper(ScalarMappable):
    """
    Map different values to colors in a colormap.

    This is useful when you wish to plot multiple series whose color corresponds to a data value, but do not want to
    use a scatter plot. This class can be instantiated in several ways:

    1. Call the class directly, providing a min and max value and (optionally) a colormap to use, e.g.::

        cm = ColorMapper(0, 10, cmap='viridis')

    2. Use the ``from_data`` method to automatically set the min and max values to those of some sort of data array,
       e.g.::

        cm = ColorMapper.from_data(np.arange(10))

    3. Use the ``from_norm`` method to specify the normalization manually. This allows the use of Matplotlib's
       `various normalizations <https://matplotlib.org/stable/tutorials/colors/colormapnorms.html>`_ with the 
       colormapper.

    4. Use the ``from_discrete_norm`` method to set up a colormapper using any color map with discrete levels.
       This can also be accomplished by passing a :class:`matplotlib.colors.BoundaryNorm` instance to ``from_norm``,
       but ``from_discrete_norm`` automatically figures out the maximum number of colors in the selected colormap.

    Either method accepts all keywords for :class:`matplotlib.cm.ScalarMappable` except ``norm`` which is set
    automatically. Calling the instance will return an RGBA tuple that can be given to the ``color`` keyword of a
    matplotlib plotting function::

        pyplot.plot(np.arange(10), color=cm(5))

    The instance of this class would then be used as the mappable for a colorbar, e.g.::

        pyplot.colorbar(cm)

    Init parameters:

    :param vmin: the bottom value for the color map.
    :type vmin: int or float

    :param vmax: the top value for the color map
    :type vmax: int or float

    :param cmap: the color map to use. May be a string that gives the name of the colormap or a Colormap instance.
    :type cmap: str or :class:`matplotlib.colors.Colormap`

    :param **kwargs: additional keyword arguments passed through to :class:`matplotlib.cm.ScalarMappable`
    """
    def __init__(self, vmin, vmax, cmap='viridis', **kwargs):
        norm = kwargs.pop('norm', Normalize(vmin=vmin, vmax=vmax))
        super(ColorMapper, self).__init__(norm=norm, cmap=cmap, **kwargs)
        # This is a necessary step for some reason. Not sure why
        self.set_array([])

    def __call__(self, value):
        return self.to_rgba(value)

    @classmethod
    def from_norm(cls, norm, cmap='viridis', **kwargs):
        """Create a color mapper from a :class:`matplotlib.colors.Normalize` subclass instance.

        Matplotlib offers `different ways to normalize the color scale <https://matplotlib.org/stable/tutorials/colors/colormapnorms.html>`_,
        including logarithmic and symmetric around zero. This method allows you to specify different normalizations
        to use with a color mapper instance.

        :param norm: the normalization to use
        :type norm: :class:`matplotlib.colors.Normalize` subclass

        :param cmap: the color map to use. May be a string that gives the name of the colormap or a Colormap instance.
        :type cmap: str or :class:`matplotlib.colors.Colormap`

        :param **kwargs: additional keyword arguments passed through to :class:`matplotlib.cm.ScalarMappable`
        """
        return cls(vmin=None, vmax=None, cmap=cmap, norm=norm, **kwargs)

    @classmethod
    def from_discrete_norm(cls, boundaries, cmap='viridis', norm_kws=dict(), **kwargs):
        """Create a color mapper that uses discrete color levels

        A :class:`matplotlib.colors.BoundaryNorm` will map data to discrete color levels pulled from a continuous color scale
        based on bins. It also requires you to specify how many colors from the color scale to use. This is inconvenient if you
        want to use the whole range, since scales have different numbers of colors. This method will automatically use the full
        range of colors. If you want to use a subset of colors, construct the :class:`~matplotlib.colors.BoundaryNorm` instance
        yourself and pass it to ``from_norm``.

        :param boundaries: A sequence of boundary edges used to map values to colors. For example, [0, 1, 2] would map values between
        0 to 1 to one color and 1 to 2 to a second.

        :param cmap: the color map to use. May be a string that gives the name of the colormap or a Colormap instance.
        :type cmap: str or :class:`matplotlib.colors.Colormap`

        :param norm_kws: additional keyword arguments passed through to :class:`matplotlib.colors.BoundaryNorm`. Note that the
        ``boundaries`` and ``ncolors`` keywords are already specified.

        :param **kwargs: additional keyword arguments passed through to :class:`matplotlib.cm.ScalarMappable`
        """
        if isinstance(cmap, str):
            cmap = get_cmap(cmap)
        return cls(vmin=None, vmax=None, cmap=cmap, norm=discrete_norm(boundaries, cmap, values_are_bounds=True, **norm_kws), **kwargs)

    @classmethod
    def from_discrete_values(cls, values, cmap='viridis', norm_kws=dict(), **kwargs):
        """Create a color mappers that has one color per given value.

        :param values:  Values that should be represented in the colorbar. Each unique value will have
         its own colors.

        Other parameters are identical to :meth:`from_discrete_norm`.
        """
        return cls(vmin=None, vmax=None, cmap=cmap, norm=discrete_norm(values, cmap, values_are_bounds=False, **norm_kws), **kwargs)

    @classmethod
    def from_data(cls, data, **kwargs):
        """
        Create a :class:`ColorMapper` instance from a data array, with min and max values set to those of the data.

        :param data: the data array. May be any type that ``numpy.min()`` and ``numpy.max()`` will correctly return a
         scalar value for.

        :param **kwargs: additional keyword args passed through to the class __init__ method.

        :return: a new class instance
        :rtype: :class:`ColorMapper`
        """
        return cls(vmin=np.nanmin(data), vmax=np.nanmax(data), **kwargs)


def discrete_norm(values, cmap='viridis', values_are_bounds=False, **norm_kws):
    """Construct a Matplotlib normalization for discrete values

    :param values: Values that should be represented in the colorbar. See ``values_are_bounds``
     for how these are interpreted.

    :param cmap: Which colormap will be used. Note that this does not automatically 
     ensure that the plot uses this colormap, but only that the discrete color map
     uses the full range of colors available.

    :param values_are_bounds: If ``False`` (the default), then the values given are assumed
     to be the exact values that should be displayed in the color map, and each will receive
     its own color. If ``True``, then these are interpreted as the borders between colors,
     with the first and last specifying the min and max values. 

    :param norm_kws: Additional keyword arguments are passed through to 
     :class:`matplotlib.colors.BoundaryNorm`.
    """
    if values_are_bounds:
        boundaries = values
    else:
        boundaries = _vals_to_boundaries(values)
    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
    return BoundaryNorm(boundaries=boundaries, ncolors=cmap.N, **norm_kws)


def _vals_to_boundaries(values):
    svalues = np.unique(values)
    boundaries = np.zeros(svalues.size + 1)
    boundaries[1:-1] = 0.5*(svalues[:-1] + svalues[1:]) 
    boundaries[0] = boundaries[1] - (boundaries[2] - boundaries[1])
    boundaries[-1] = boundaries[-2] + (boundaries[-2] - boundaries[-3])
    return boundaries
