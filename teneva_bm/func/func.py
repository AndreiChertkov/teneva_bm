import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from teneva_bm import Bm


class Func(Bm):
    def __init__(self, d=7, n=16, seed=42, name=None):
        super().__init__(d, n, seed, name)

    @property
    def is_func(self):
        return True

    @property
    def opts_plot(self):
        return {'dy_min': 5., 'dy_max': 0.}

    @property
    def ref_i(self):
        return np.array([5, 3, 9, 11, 14, 3, 10], dtype=int)

    @property
    def with_plot(self):
        return self.d == 2

    def plot(self, fpath=None):
        if self.d != 2:
            raise ValueError('Plot is supported only for 2D case')

        X = [np.linspace(self.a[k], self.b[k], self.n[k]) for k in range(2)]
        X1, X2 = np.meshgrid(*X)
        X = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)])

        Y = self.get_poi(X, skip_process=True)
        Y = Y.reshape(X1.shape)

        y_min = np.min(Y)
        y_max = np.max(Y)

        if self.y_min_real is not None:
            y_min = min(y_min, self.y_min_real)

        if self.y_max_real is not None:
            y_max = max(y_max, self.y_max_real)

        y_min_func = y_min
        y_max_func = y_max
        y_dlt_func = y_max_func - y_min_func

        y_min -= self.opts_plot['dy_min']
        y_max += self.opts_plot['dy_max']

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        surface = ax.plot_surface(X1, X2, Y, cmap='coolwarm',
            linewidth=4, antialiased=True)

        ax.contourf(X1, X2, Y, zdir='z', cmap='coolwarm',
            offset=y_min, alpha=0.7, zorder=1000)

        if self.x_max_real is not None:
            x1, x2 = self.x_max_real
            ax.scatter([x1], [x2], [y_min],
                s=150, c='r', marker='o', zorder=100)

        if self.x_min_real is not None:
            x1, x2 = self.x_min_real
            ax.scatter([x1], [x2], [y_min],
                s=150, c='b', marker='*', zorder=100)

        ax.set_xlabel('X1', labelpad=25)
        ax.set_ylabel('X2', labelpad=35)

        ax.set_xlim(self.a[0], self.b[0])
        ax.set_ylim(self.a[1], self.b[1])
        ax.set_zlim(y_min, y_max)

        ax.set_xticks(np.linspace(self.a[0], self.b[0], 5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='x', which='major', pad=-2,
            labelrotation=+50, labelright=True)

        ax.set_yticks(np.linspace(self.a[1], self.b[1], 5))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='y', which='major', pad=15,
            labelrotation=-25, labelright=True, labelleft=False)

        ax.set_zticks(np.linspace(y_min_func, y_max_func, 5))
        if abs(y_min) > 1.E+4 or abs(y_max) > 1.E+4:
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.1e'))
            ax.tick_params(axis='z', which='major', pad=15)
        elif y_dlt_func < 10:
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.tick_params(axis='z', which='major', pad=5)
        else:
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.tick_params(axis='z', which='major', pad=5)

        name = self.name
        if 'Func' in name:
            name = name.split('Func')[1]
        name += '\nfunction'
        fig.text(0.70, 0.88, name, transform=ax.transAxes,
            style='normal', weight='bold', fontsize=18, color='#61677A')

        if fpath:
            fpath = self.path_build(fpath, 'png')
            plt.savefig(fpath, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
