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

    def plot(self, colors=None, linestyles=None, linewidths=None, markers=None, y_lim=(0, 1),
             save_path=None, coordinate_name=None, legend_loc='lower right', sub=None, s=False):

        ax = plt.gca()
        labels = self.labels
        n = len(labels)
        line_lists = [None]*n

        assert len(colors) == n

        assert len(linestyles) == n

        if linewidths is None:
            linewidths = [1.5 for i in range(n)]
        else:
            assert len(linewidths) == n

        assert len(markers) == n


        for i, lb in enumerate(labels):
            plt.subplot(sub)
            plt.ylim(y_lim)
            line_lists[i], = plt.plot(list(range(self.epochs)),
                                      self.history[lb],
                                      color=colors[i],
                                      label=lb,
                                      linestyle=linestyles[i],
                                      marker=markers[i],
                                      linewidth=linewidths[i],
                                      markersize=7)
            # plt.errorbar(list(range(self.epochs)),self.history[lb],yerr=self.err[lb],elinewidth=1,capsize=4, ecolor=colors[i],color=colors[i])
        ax = plt.gca()
        fontsize = 27
        # if sub==131:
        #     ax.legend(tuple(line_lists), labels, loc='best', fontsize=fontsize, bbox_to_anchor=(3.1,-0.26),ncol=6)
        # ax.legend(tuple(line_lists), labels, loc='lower center',fontsize=10.5,ncol=3)
        # ax.legend(tuple(line_lists), labels, fontsize=9.5, loc=legend_loc,ncol=2)
        if coordinate_name is not None:
            if sub==131:
                plt.xlabel(coordinate_name[0], fontsize=fontsize)
                plt.ylabel(coordinate_name[1], fontsize=fontsize)
            else:
                plt.xlabel(coordinate_name[0], fontsize=fontsize)

        # ax.xaxis.set_major_locator(plt.FixedLocator([0,1,2,4]))
        #plt.xticks(range(0,7,1),('0','10x1','5x5','10x10','15x15','20x20','25x25'),rotation=30)
        plt.tick_params(labelsize=fontsize)

        if sub == 131 or sub == 132:
            plt.xticks(range(0, 6, 1), ['50','60','70','80','90','100'])
        else:
            plt.xticks(range(0, 6, 1), ['500', '600', '700', '800', '900', '1000'])

        # else:
        #     plt.xticks(range(0, 11, 1), ['50', '55', '60', '65', '70', '75', '80', '85', '90', '95', '200'])

        plt.subplots_adjust(left=0.045, right=0.99, top=0.97, bottom=0.32)
        plt.subplots_adjust(wspace=0.10, hspace=0.25)
        plt.grid(linestyle=':')


        if s:
            save_path = os.path.expanduser(save_path)
            plt.savefig(save_path)

