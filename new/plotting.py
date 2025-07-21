import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator, LogFormatterSciNotation,  LogFormatter, NullFormatter


class Hist:
    """1d histogram class with options for linear, log or x^3 binning. Entries can be weighted. 
        Plotting can be cumulative / survival. 
    """
    def __init__(self, nbins: int, tmax: float, nbins_factor: int = 1,
                 xlog: bool = False, x3bin: bool = False):
        """initialize the histogram

        Args:
            nbins (int): Number of bins. 
            tmax (float): Upper limit for binning. Can also be Emax. 
            nbins_factor (int, optional): Multiply bins for finer resolution. Defaults to 1.
            xlog (bool, optional): If True, the binning is logarithmic. Defaults to False.
            x3bin (bool, optional): If True, use bins linear in (E/Emax)^{1/3}. Defaults to False.
        """
        self.nbins = nbins * nbins_factor
        self.tmax = tmax  # For x3bin, tmax is Emax!
        self.xlog = xlog
        self.x3bin = x3bin

        if xlog:
            self.bins = np.logspace(np.log10(1e-3), np.log10(tmax), self.nbins + 1)
        elif x3bin:
            # bins are linear in cube-root space, so edges in [0,1]
            self.bins = np.linspace(0, 1, self.nbins + 1)
        else:
            self.bins = np.linspace(0, tmax, self.nbins + 1)

        self.entries = np.zeros(self.nbins, dtype=float)
        self.errors = np.zeros(self.nbins, dtype=float)
        self.hits = np.zeros(self.nbins, dtype=int)

    def add_to_bin(self, t: float, weight: float = 1.0):
        """add a data point with a certain weight to the histogram

        Args:
            t (float): Value to bin, typically t or E.
            weight (float, optional): Weight of the binned value. Defaults to 1.0.
        """
        if self.x3bin:
            # transform: E → (E/Emax)^{1/3}
            t = (t / self.tmax) ** (1/3)
        if t <= 0 or t > 1 if self.x3bin else t > self.tmax:
            return
        bin_idx = np.digitize(t, self.bins) - 1
        if 0 <= bin_idx < self.nbins:
            self.entries[bin_idx] += weight
            self.errors[bin_idx] += weight ** 2
            self.hits[bin_idx] += 1

    def reset(self):
        """reset the entire histogram
        """
        self.entries[:] = 0
        self.errors[:] = 0
        self.hits[:] = 0

    @staticmethod
    def compute_cumulative_errors(errors: np.array, kind: str="cumulative"):
        """Computed propagated errors for cumulative or survival distributions. 

        Args:
            errors (np.array): array with errors
            kind (str, optional): 'cumulative' (sum from left) or 'survival' (sum from right). Defaults to "cumulative".

        Raises:
            ValueError: kind must be 'cumulative' or 'survival'
        Returns:
            np.array: square root of cumulative sums.
        """
        errors = np.asarray(errors)
        if kind == "cumulative":
            cum_vars = np.cumsum(errors)
        elif kind == "survival":
            cum_vars = np.cumsum(errors[::-1])[::-1]
        else:
            raise ValueError("kind must be 'cumulative' or 'survival'")
        return np.sqrt(cum_vars)

    @staticmethod
    def plot_histograms(histograms, filename, xlabel, NeV,
                        ylabel="Entries", title="Histogram",
                        cumulative=False, survival=False, ylog=False, xlog=None,
                        labels=None, show=False, normalize_energy_axis=False,
                        printer=False, styles=None, colors=None):
        """Plot one or multiple 1D histograms with options for cumulative and survival plots.

        Args: 
            histograms (list[Hist]): List of Hist objects.
            filename (str): Path to save the PDF.
            xlabel (str): x-axis label.
            NeV (float): Normalization factor (number of events).
            ylabel (str): y-axis label.
            title (str): Plot title.
            cumulative (bool or list[bool]): Whether to plot cumulatively.
            survival (bool or list[bool]): Whether to plot survival (1 - CDF).
            ylog (bool): Use log scale for y.
            xlog (bool): Use log scale for x.
            labels (list[str]): Legend labels.
            show (bool): Show interactive plot.
            normalize_energy_axis (bool): Normalize x by Emax.
            printer (bool): If True, print bin table to console.
            styles (list[str]): Line styles.
            colors (list[str]): Colors.
        """

        if xlog is None:
            xlog = histograms[0].xlog

        if labels is None:
            labels = [f"Hist {i}" for i in range(len(histograms))]

        if styles is None:
            styles = ["solid"] * len(histograms)
        if colors is None:
            pastel_colors = [
                "#1BB7F0", "#EE8027", "#79FF79", "#FF6961",
                "#FD9EFD", "#EBEB4C", "#FC6595", "#005763"
            ]
            colors = pastel_colors[:len(histograms)]

        if not isinstance(cumulative, list):
            cumulative = [cumulative] * len(histograms)
        if not isinstance(survival, list):
            survival = [survival] * len(histograms)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.figure(figsize=(10, 6))

        for i, hist in enumerate(histograms):
            bin_edges = hist.bins
            if hist.x3bin:
                # convert cube-root bin edges back to physical E/Emax
                bin_edges = bin_edges ** 3

            bin_widths = bin_edges[1:] - bin_edges[:-1]
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            counts = hist.entries.copy()
            errors = hist.errors.copy()

            if not cumulative[i]:
                counts = counts / bin_widths
                norm_errors = errors / (bin_widths ** 2)
                norm_errors = np.sqrt(norm_errors)
            else:
                if survival[i]:
                    centers += 0.5 * bin_widths
                    counts = 1.0 - np.cumsum(counts) / NeV
                else:
                    counts = np.cumsum(counts) / NeV
                    errors = np.cumsum(errors) / NeV

            plt.scatter(centers, counts,
                        label=labels[i],
                        linestyle=styles[i],
                        color=colors[i],
                        linewidth=2,
                        s=30)

            if printer:
                Hist.print_table(centers, counts)

        ax = plt.gca()

        if ylog: #comment out when needed
            plt.yscale("log")
            """ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15)) 
            ax.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=15))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which='both', axis='y')"""
        else:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(axis='y', style='plain')

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.ylim(-0.1, 1.1)

        if ylog:
            plt.yscale('log')
        if xlog:
            plt.xscale('log')

        plt.savefig(filename, format="pdf", dpi=300)
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_histograms_energy(histograms, filename, xlabel, NeV,
                               ylabel="Entries", title="Histogram",
                               cumulative=False, survival=False, ylog=False,
                               xlog=None, labels=None, show=False,
                               normalize_energy_axis=False, printer=False,
                               styles=None, colors=None):
        """Variant of plot_histograms for physical energy axis,
        useful when using cube-root binning.

        Args: see plot_histogram for details. 
        """

        if xlog is None:
            xlog = histograms[0].xlog

        if labels is None:
            labels = [f"Hist {i}" for i in range(len(histograms))]

        if styles is None:
            styles = ["solid"] * len(histograms)
        if colors is None:
            pastel_colors = [
                "#1BB7F0", "#EE8027", "#79FF79", "#FF6961",
                "#FD9EFD", "#EBEB4C", "#FC6595", "#005763"
            ]
            colors = pastel_colors[:len(histograms)]

        if not isinstance(cumulative, list):
            cumulative = [cumulative] * len(histograms)
        if not isinstance(survival, list):
            survival = [survival] * len(histograms)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.figure(figsize=(10, 6))

        for i, hist in enumerate(histograms):
            if hist.x3bin:
                # Linear bins in x
                x_edges = hist.bins
                # Convert to physical E
                bin_edges = hist.tmax * (x_edges ** 3)
            else:
                bin_edges = hist.bins

            bin_widths = bin_edges[1:] - bin_edges[:-1]
            centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            counts = hist.entries.copy()
            errors = hist.errors.copy()

            if not cumulative[i]:
                counts = counts / (bin_widths * NeV)
                norm_errors = errors / (bin_widths ** 2 * NeV ** 2)
                norm_errors = np.sqrt(norm_errors)
            else:
                if survival[i]:
                    centers += 0.5 * bin_widths
                    counts = 1.0 - np.cumsum(counts) / NeV
                else:
                    counts = np.cumsum(counts) / NeV
                    errors = np.cumsum(errors) / NeV

            if normalize_energy_axis:
                centers /= bin_edges[-1]

            plt.scatter(
                centers, counts,
                label=labels[i],
                color=colors[i],
                s=30
            )

            if printer:
                Hist.print_table(centers, counts)

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True)
        plt.legend()

        ax = plt.gca()

        if ylog: #comment out when needed
            plt.yscale("log") 
            """ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
            ax.yaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=15))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which='both', axis='y')"""
        else:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(axis='y', style='plain')

        if ylog:
            plt.yscale('log')
        if xlog:
            plt.xscale('log')

        plt.savefig(filename, format = "pdf", dpi = 300)
        if show:
            plt.show()
        plt.close()


    @staticmethod
    def print_table(centers: np.array, y_vals: np.array):
        """print table of bin centers and values

        Args:
            centers (np.array): centers to print
            y_vals (np.array): values to print
        """
        print(f"{'S(t)':>20} | {'t':>20}")
        print("-" * 45)
        for c, y in zip(centers, y_vals):
            print(f"{c:>20} | {y:>20}")


def save_unique_plot(filepath: str, directory: str) -> str:
    """save a plot with a unique filename by appending _1, _2, etc.

    Args:
        filepath (str): desired filename
        directory (str): where to store the file

    Returns:
        str: unique filename
    """
    base = os.path.basename(filepath)
    name, ext = os.path.splitext(base)
    candidate = os.path.join(directory, base)
    count = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{name}_{count}{ext}")
        count += 1
    return candidate


def clear_directory(directory: str):
    """delete all files from that directory

    Args:
        directory (str): target directory
    """
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if os.path.isfile(path):
            os.remove(path)

class Hist2D:
    """Simple 2D histogram for density and conditional expectation plots.
    """
    def __init__(self, nbins_x: int, nbins_y: int, xmax: float, ymax: float):
        """Initialize a 2D histogram.

        Args:
            nbins_x (int): number of bins in x direction.
            nbins_y (int): number of bins in y direction.
            xmax (float): maximum x.
            ymax (float): maximum y.
        """
        self.nbins_x = nbins_x
        self.nbins_y = nbins_y
        self.xmax = xmax
        self.ymax = ymax

        self.xbins = np.linspace(0, xmax, nbins_x + 1)
        self.ybins = np.linspace(0, ymax, nbins_y + 1)

        self.entries = np.zeros((nbins_x, nbins_y), dtype=float)

    def add(self, x: float, y: float, weight: float = 1.0):
        """add a value with certain weight to the 2d histogram.

        Args:
            x (float): x value
            y (float): y value
            weight (float, optional): Weight of the value to be binned. Defaults to 1.0.
        """
        if x < 0 or x > self.xmax or y < 0 or y > self.ymax:
            return
        ix = np.digitize(x, self.xbins) - 1
        iy = np.digitize(y, self.ybins) - 1
        if 0 <= ix < self.nbins_x and 0 <= iy < self.nbins_y:
            self.entries[ix, iy] += weight

    def plot_density(self, filename: str, xlabel: str="x", ylabel: str="y", show: bool=False):
        """plot the 2d density histogram as a heatmap.

        Args:
            filename (str): filename of the plot
            xlabel (str, optional): x label. Defaults to "x".
            ylabel (str, optional): y label. Defaults to "y".
            show (bool, optional): whether to show the plot to the console. Defaults to False.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        X, Y = np.meshgrid(self.xbins, self.ybins, indexing='ij')
        c = ax.pcolormesh(X, Y, self.entries, shading='auto', cmap='viridis')
        fig.colorbar(c, ax=ax, label='Entries')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        #ax.set_title(title)
        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        if show:
            plt.show()
        plt.close()

    def plot_expectation(self, filename: str, xlabel: str="t", ylabel: str=r"t_E", show: bool=False):
        """        Plot conditional expectations ⟨Y|X⟩ and ⟨X|Y⟩.

        Args:
            filename (str): filename of the plot
            xlabel (str, optional): x label. Defaults to "t".
            ylabel (str, optional): y label. Defaults to "t_E".
            show (bool, optional): whether to show the plot to the console. Defaults to False.
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # ⟨ y | x ⟩ vs x  → here y = tE, x = t
        means_y_given_x = []
        xs = []
        for ix in range(self.nbins_x):
            slice_ = self.entries[ix, :]
            total = slice_.sum()
            if total > 0:
                y_centers = 0.5 * (self.ybins[:-1] + self.ybins[1:])
                mean = np.sum(slice_ * y_centers) / total
                means_y_given_x.append(mean)
                x_center = 0.5 * (self.xbins[ix] + self.xbins[ix+1])
                xs.append(x_center)
        ax.plot(xs, means_y_given_x, 'r-', lw=2, label=r"$\langle t_E | t \rangle$")

        # ⟨ x | y ⟩ vs y  → here x = t, y = tE
        means_x_given_y = []
        ys = []
        for iy in range(self.nbins_y):
            slice_ = self.entries[:, iy]
            total = slice_.sum()
            if total > 0:
                x_centers = 0.5 * (self.xbins[:-1] + self.xbins[1:])
                mean = np.sum(slice_ * x_centers) / total
                means_x_given_y.append(mean)
                y_center = 0.5 * (self.ybins[iy] + self.ybins[iy+1])
                ys.append(y_center)
        ax.plot(means_x_given_y, ys, 'b-', lw=2, label=r"$\langle t | t_E \rangle$")

        # Identity line if square domain
        if abs(self.xmax - self.ymax) / self.xmax < 0.1:
            minmax = min(self.xmax, self.ymax)
            ax.plot([0, minmax], [0, minmax], 'k--', lw=1, label="Identity")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        #ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        if show:
            plt.show()
        plt.close()