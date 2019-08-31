"""
Function to create subplots from a ``pd.DataFrame`` object and a series of
matplotlib parameters.
"""
import matplotlib.pyplot as plt

# pylint: disable=too-many-locals, invalid-name


def create_subplot(**kwargs):
    """ Same as module DOCSTRING. """

    ax = kwargs["ax"]
    xaxis = kwargs["xaxis"]
    yaxis = kwargs["yaxis"]
    df = kwargs["df"]
    ylabel = kwargs["ylabel"]
    column_total = kwargs["column_total"]
    color_index = kwargs["color_index"]
    num_colors = kwargs["num_colors"]
    xlabel = kwargs["xlabel"]
    y_axis_label_size = kwargs["y_axis_label_size"]
    x_axis_label_size = kwargs["x_axis_label_size"]
    legend_size = kwargs["legend_size"]
    tick_label_size = kwargs["tick_label_size"]
    axis_font = kwargs["axis_font"]
    legend_font = kwargs["legend_font"]
    text_opacity = kwargs["text_opacity"]
    xaxis_opacity = kwargs["xaxis_opacity"]
    column_count = kwargs["column_count"]

    # ===PLOT===
    graph = df.plot(x=xaxis, y=yaxis, ax=ax, use_index=True)
    # legend=True)
    # john
    plt.legend(loc="best")

    # hacks
    # ax.set_ylim(top=0.93, bottom=0.4)
    # ax.set_xlim(left=-0.1, right=10.5)

    MARKERS = [
        ".",
        ",",
        "o",
        "v",
        "s",
        "p",
        "P",
        "H",
        "+",
        "x",
        "X",
        "D",
        "d",
        "|",
        "_",
        "<",
        ">",
        "^",
        "8",
        "*",
        "h",
        "1",
        "2",
        "3",
        "4",
    ]

    # distinct line colors/styles for many lines
    # LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    LINE_STYLES = ["solid"]
    num_styles = len(LINE_STYLES)

    use_markers = False
    if use_markers:
        assert len(MARKERS) >= column_count
    colormap = plt.get_cmap("magma")  #'gist_rainbow'

    j = 0
    while color_index < column_total:
        plt.gca().get_lines()[j].set_color(
            colormap(color_index // num_styles * float(num_styles) / num_colors)
        )

        if use_markers:
            plt.gca().get_lines()[j].set_marker(MARKERS[j])
            # plt.gca().get_lines()[j].set_linestyle(LINE_STYLES[i%num_styles])
            plt.gca().get_lines()[j].set_markersize(7.0)

        plt.gca().get_lines()[j].set_linewidth(3.0)
        color_index += 1
        j += 1

    # add axis labels
    plt.xlabel(
        xlabel, fontproperties=axis_font, fontsize=y_axis_label_size, alpha=text_opacity
    )
    plt.ylabel(
        ylabel, fontproperties=axis_font, fontsize=y_axis_label_size, alpha=text_opacity
    )

    # change font of legend
    legend = graph.legend(prop={"size": legend_size})
    plt.setp(legend.texts, fontproperties=legend_font, alpha=text_opacity)

    # set size of tick labels
    graph.tick_params(axis="both", which="major", labelsize=tick_label_size)

    # set fontname for tick labels
    for tick in graph.get_xticklabels():
        tick.set_fontname("DecimaMonoPro")
    for tick in graph.get_yticklabels():
        tick.set_fontname("DecimaMonoPro")

    # set color for tick labels
    _ = [t.set_color("#303030") for t in ax.xaxis.get_ticklabels()]
    _ = [t.set_color("#303030") for t in ax.yaxis.get_ticklabels()]

    # create bolded x-axis
    graph.axhline(y=0, color="black", linewidth=1.3, alpha=xaxis_opacity)  # 0

    # Set color of subplots.
    ax.set_facecolor("#F0F0F0")

    return graph, color_index
