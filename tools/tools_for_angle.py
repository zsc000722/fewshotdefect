import matplotlib.pyplot as plt
import os


class TrainingHistory(object):
    def __init__(self, labels):
        self.l = len(labels)
        self.history = {}
        self.err = {}
        self.epochs = 0
        self.err_epochs = 0
        self.labels = labels
        for lb in self.labels:
            self.history[lb] = []
        for lb in self.labels:
            self.err[lb] = []

    def update(self, values):
        assert len(values) == self.l
        for i, v in enumerate(values):
            label = self.labels[i]
            self.history[label].append(v)
        self.epochs += 1

    def err_update(self, values):
        assert len(values) == self.l
        for i, v in enumerate(values):
            label = self.labels[i]
            self.err[label].append(v)
        self.err_epochs += 1

    def plot(self, epoch=None, colors=None, linestyles=None, linewidths=None, markers=None, y_lim=(0, 1),
             save_path=None, coordinate_name=None, legend_loc='lower right', ):

        ax = plt.gca()
        labels = self.labels
        n = len(labels)
        line_lists = [None] * n

        assert len(colors) == n

        assert len(linestyles) == n

        if linewidths is None:
            linewidths = [1.5 for i in range(n)]
        else:
            assert len(linewidths) == n

        assert len(markers) == n

        plt.ylim(y_lim)

        self.epochs = epoch

        for i, lb in enumerate(labels):
            line_lists[i], = plt.plot(self.epochs,
                                      self.history[lb],
                                      color=colors[i],
                                      label=lb,
                                      linestyle=linestyles[i],
                                      marker=markers[i],
                                      linewidth=linewidths[i],
                                      markersize=7)
            # plt.errorbar(list(range(self.epochs)),self.history[lb],yerr=self.err[lb],elinewidth=0.5,capsize=2.5, ecolor=colors[i],color=colors[i])

        # ax.legend(tuple(line_lists), labels, loc='best', fontsize=11.5, bbox_to_anchor=(1.018, 1.2),ncol=4)
        ax.legend(tuple(line_lists), labels, loc=legend_loc, fontsize=25, ncol=1)

        if coordinate_name is not None:
            plt.xlabel(coordinate_name[0], fontsize=25, weight='bold')
            plt.ylabel(coordinate_name[1], fontsize=25, weight='bold')

        plt.tick_params(labelsize=25)
        plt.xticks(self.epochs, fontsize=25, weight='bold')
        plt.yticks(fontsize=25, weight='bold')
        plt.subplots_adjust(left=0.1, right=0.985, bottom=0.113, top=0.95)
        plt.grid(linestyle=':')

        save_path = os.path.expanduser(save_path)
        plt.savefig(save_path)
