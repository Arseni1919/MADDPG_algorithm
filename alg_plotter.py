from alg_constrants_amd_packages import *
from alg_logger import run


class ALGPlotter:
    def __init__(self):
        if PLOT_LIVE:
            self.fig, _ = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
            self.actor_losses = []
            self.critic_losses_1 = []
            self.critic_losses_2 = []

    def plot_online(self, graph_dict):
        # plot live:
        if PLOT_LIVE:
            def plot_graph(ax, indx, list_of_values, label, color='r'):
                ax[indx].cla()
                ax[indx].plot(list(range(len(list_of_values))), list_of_values, c=color)  # , edgecolor='b')
                ax[indx].set_title(f'Plot: {label}')
                ax[indx].set_xlabel('iters')
                ax[indx].set_ylabel(f'{label}')

            ax = self.fig.get_axes()
            b_indx = graph_dict['b_indx']

            self.actor_losses.append(graph_dict['actor_loss'])
            self.critic_losses_1.append(graph_dict['critic_loss_1'])
            self.critic_losses_2.append(graph_dict['critic_loss_2'])

            # graphs
            if b_indx % 9 == 0:
                plot_graph(ax, 1, self.actor_losses, 'actor_loss')
                plot_graph(ax, 2, self.critic_losses_1, 'critic_loss_1')
                plot_graph(ax, 2, self.critic_losses_2, 'critic_loss_2')
                # plot_graph(ax, 4, graph_dict['critic_w_mse'], 'critic_w_mse')

                plt.pause(0.05)

    def plot_summary(self):
        pass

    @staticmethod
    def neptune_update(loss):
        if NEPTUNE:
            run['acc_loss'].log(loss)
            run['acc_loss_log'].log(f'{loss}')


plotter = ALGPlotter()




