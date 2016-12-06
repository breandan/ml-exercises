def pop_all():
    #bring all the figures hiding in the background to the foreground
    all_figures=[manager.canvas.figure for manager in matplotlib.\
        _pylab_helpers.Gcf.get_all_fig_managers()]
    [fig.canvas.manager.show() for fig in all_figures]
    return len(all_figures)