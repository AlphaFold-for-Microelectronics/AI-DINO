import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

props = fm.FontProperties(family=['Lato', 'sans-serif'], size='large')

plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.linewidth'] = 1

def format_axis(ax, xlabel='', ylabel='', title='', xbins=None, ybins=None):
    """
    Apply consistent formatting to a matplotlib axis including fonts and tick settings.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axis object to format
    xlabel : str, optional
        Label for the x-axis. Default is empty string.
    ylabel : str, optional  
        Label for the y-axis. Default is empty string.
    title : str, optional
        Title for the axis. Default is empty string.
    xbins : int, optional
        Approximate number of tick marks on x-axis. If provided, attempts to set
        the tick locator to use this many bins. Falls back to numticks if nbins fails.
    ybins : int, optional
        Approximate number of tick marks on y-axis. If provided, attempts to set
        the tick locator to use this many bins. Falls back to numticks if nbins fails.
    """
    
    ax.set_xlabel(xlabel, fontproperties=props)
    ax.set_ylabel(ylabel, fontproperties=props)
    ax.set_title(title, fontproperties=props)
    ax.xaxis.offsetText.set_fontproperties(props)
    ax.yaxis.offsetText.set_fontproperties(props)
    
    for label in ax.get_xticklabels(which='both'):
        label.set_fontproperties(props)
    for label in ax.get_yticklabels(which='both'):
        label.set_fontproperties(props)
        
    if xbins:
        try: ax.locator_params(axis='x', nbins=xbins)
        except: ax.locator_params(axis='x', numticks=xbins+1)
            
    if ybins:
        try: ax.locator_params(axis='y', nbins=ybins)
        except: ax.locator_params(axis='y', numticks=ybins+1)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Create a new colormap from a subset of an existing colormap.
    
    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        The original colormap to truncate
    minval : float, optional
        Starting point in the original colormap (0.0 to 1.0). Default is 0.0.
    maxval : float, optional
        Ending point in the original colormap (0.0 to 1.0). Default is 1.0.
    n : int, optional
        Number of discrete colors to sample from the specified range. Default is 100.
        
    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        New colormap
    
    References
    ----------
    Adapted from: https://gist.github.com/salotz/4f585aac1adb6b14305c
    """
    
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(
        n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def calculate_figsize_with_colorbar(num_rows, num_cols, subplot_size=3.5, wspace=0.15, hspace=0.15,
                                    cbar_ratio=0.05, cbar_orientation='vertical'):
    """
    Calculate correct figsize for subplots with colorbar(s).
    
    Parameters:
    -----------
    num_rows, num_cols : int
        Total number of subplot rows/columns including the colorbar
    subplot_size : float
        Desired size of each main data subplot
    wspace, hspace : float
        Relative spacing passed to subplots_adjust
    cbar_ratio : float
        Size ratio of colorbar relative to main subplots
    cbar_orientation : str
        'vertical' (colorbar is a column) or 'horizontal' (colorbar is a row)
    
    Returns:
    --------
    figsize : tuple
        (width, height) for plt.figure(figsize=...)
    width_ratios : list or None
        Width ratios for gridspec_kw (if vertical colorbar)
    height_ratios : list or None  
        Height ratios for gridspec_kw (if horizontal colorbar)
    """
    
    if cbar_orientation == 'vertical':
        # Colorbar takes up one column
        num_data_cols = num_cols - 1
        num_data_rows = num_rows
        width_ratios = [1] * num_data_cols + [cbar_ratio]
        height_ratios = None
        
        # Width calculation with ratios
        total_width_ratio = num_data_cols + cbar_ratio
        target_available_width = subplot_size * total_width_ratio
        
        avg_subplot_width = target_available_width / num_cols
        total_spacing_width = wspace * avg_subplot_width * (num_cols - 1)

        total_content_width = target_available_width + total_spacing_width
        figsize_width = total_content_width / 0.88
        
        # Height calculation
        target_height = subplot_size * num_data_rows
        total_spacing_height = hspace * subplot_size * (num_data_rows - 1)
        total_content_height = target_height + total_spacing_height  
        figsize_height = total_content_height / 0.88
        
    else:
        # Colorbar takes up one row
        num_data_cols = num_cols  
        num_data_rows = num_rows - 1
        width_ratios = None
        height_ratios = [cbar_ratio] + [1] * num_data_rows
        
        # Width calculation
        target_width = subplot_size * num_data_cols
        total_spacing_width = wspace * subplot_size * (num_data_cols - 1)
        total_content_width = target_width + total_spacing_width
        figsize_width = total_content_width / 0.88
        
        # Height calculation with ratios
        total_height_ratio = num_data_rows + cbar_ratio
        target_available_height = subplot_size * total_height_ratio
        
        avg_subplot_height = target_available_height / num_rows
        total_spacing_height = hspace * avg_subplot_height * (num_rows - 1)
        
        total_content_height = target_available_height + total_spacing_height
        figsize_height = total_content_height / 0.88
    
    return (figsize_width, figsize_height), width_ratios, height_ratios

def create_figure_with_colorbar(num_rows, num_cols, subplot_size=3.5, wspace=0.15, hspace=0.15,
                                cbar_ratio=0.05, cbar_orientation='vertical', sharex=None, sharey=None):
    """
    Create figure with consistent subplot sizing with colorbar(s).
    
    Parameters:
    -----------
    num_rows, num_cols : int
        Total number of subplot rows/columns including the colorbar
    subplot_size : float
        Desired size of each main data subplot
    wspace, hspace : float
        Relative spacing passed to subplots_adjust
    cbar_ratio : float
        Size ratio of colorbar relative to main subplots
    cbar_orientation : str
        'vertical' (colorbar is a column) or 'horizontal' (colorbar is a row)
    sharex, sharey: str
        Whether to share the x/y axes across subplots
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    axes : numpy.array
        Array of all subplot axes
    """
    
    # Calculate correct figsize and ratios
    figsize, width_ratios, height_ratios = calculate_figsize_with_colorbar(
        num_rows, num_cols, subplot_size, wspace, hspace, cbar_ratio, cbar_orientation
    )
    
    # Create subplots
    gridspec_kw = {}
    if width_ratios is not None:
        gridspec_kw['width_ratios'] = width_ratios
    if height_ratios is not None:
        gridspec_kw['height_ratios'] = height_ratios
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, gridspec_kw=gridspec_kw, sharex=sharex, sharey=sharey)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    
    return fig, axes