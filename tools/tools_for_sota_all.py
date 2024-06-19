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
        # if sub==235:
        #     ax.legend(tuple(line_lists), labels, loc='best', fontsize=20, bbox_to_anchor=(2.5,-0.30),ncol=4)
        ax = plt.gca()
        fontsize = 30
        if sub==234:
            ax.legend(tuple(line_lists), labels, loc='best', fontsize=27, bbox_to_anchor=(3.23,-0.37),ncol=7)
        # ax.legend(tuple(line_lists), labels,  fontsize=18, loc='lower center',ncol=4)
        # ax.legend(tuple(line_lists), labels, fontsize=9.5, loc=legend_loc,ncol=2)

        # font = {'style': 'normal',
        #          'family': 'Times New Roman',
        #          'weight': 'normal',
        #          'size': 30,
        #          }
        plt.xlabel(coordinate_name[0], fontsize=fontsize)
        if sub == 231 or sub == 234:
            plt.ylabel(coordinate_name[1], fontsize=fontsize)
        # if coordinate_name is not None:
        #     if sub==231 or sub==234:
        #         plt.xlabel(coordinate_name[0], fontsize=24)
        #         plt.ylabel(coordinate_name[1], fontsize=24)
        #     else:
        #         plt.xlabel(coordinate_name[0], fontsize=24)

        # ax.xaxis.set_major_locator(plt.FixedLocator([0,1,2,4]))
        #plt.xticks(range(0,7,1),('0','10x1','5x5','10x10','15x15','20x20','25x25'),rotation=30)
        plt.tick_params(labelsize=fontsize)


        if sub == 231 or sub == 232:
            plt.xticks(range(0, 6, 1), ['50','60','70','80','90','100'])
        if sub == 233:
            plt.xticks(range(0, 6, 1), ['500', '600', '700', '800', '900', '1000'])
        if sub == 234 or sub == 235:
            plt.xticks(range(0, 11, 1), ['50', '', '60', '', '70', '', '80', '', '90', '', '100'])
        if sub == 236:
            plt.xticks(range(0, 11, 1), ['500', '', '600', '', '700', '', '800', '', '900', '', '1000'])


        plt.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.2)
        plt.subplots_adjust(wspace=0.1, hspace=0.42)
        plt.grid(linestyle=':')


        if s:
            save_path = os.path.expanduser(save_path)
            plt.savefig(save_path)

