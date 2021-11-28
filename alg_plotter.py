from alg_GLOBALS import *
import logging
"""
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
"""


class ALGPlotter:
    """
    This object is responsible for plotting, logging and neptune updating.
    """
    def __init__(self, plot_life=True, plot_neptune=False, name='nameless', tags=None, plot_per=1, agents=None):

        self.plot_life = plot_life
        self.plot_neptune = plot_neptune
        self.name = name
        self.tags = [] if tags is None else tags
        self.plot_per = plot_per
        self.agents = agents

        self.run = {}
        self.neptune_initiated = False

        if self.plot_life:
            self.fig = plt.figure(figsize=plt.figaspect(.3))
            self.fig.suptitle(f'{name}')
            self.fig.tight_layout()
            # ax_1 = fig.add_subplot(1, 1, 1, projection='3d')
            # self.ax_1 = self.fig.add_subplot(1, 5, 1, projection='3d')
            self.ax_2 = self.fig.add_subplot(1, 4, 1)
            self.ax_3 = self.fig.add_subplot(1, 4, 2)
            self.ax_4 = self.fig.add_subplot(1, 4, 3)
            self.ax_5 = self.fig.add_subplot(1, 4, 4)

            self.means_list = {agent: [] for agent in self.agents}
            self.stds_list = {agent: [] for agent in self.agents}
            self.losses_list_actor = {agent: [] for agent in self.agents}
            self.loss_list_critic = []
            self.total_scores = [0]
            self.total_avg_scores = [0]
            self.state_means = {agent: [] for agent in self.agents}
            self.state_stds = {agent: [] for agent in self.agents}
            self.list_state_mean_2 = []
            self.list_state_std_2 = []

        print(colored(f'~[INFO]: "ALGPlotter instance created."', 'green'))

    def plot(self, iteration, average_result_dict, loss_critic,
             means, stds, losses_actor, state_stats
             # actor_output_tensor, observations_tensor,
             # scores, avg_scores, state_stat_mean, state_stat_std
             ):
        if self.plot_life:
            average_scores = np.array(list(average_result_dict.values()))
            self.total_scores.append(average_scores.mean())
            self.total_avg_scores.append(self.total_avg_scores[-1] * 0.9 + np.mean(average_scores) * 0.1)
            # PLOT
            # mean_list.append(actor_output_tensor.mean().detach().squeeze().item())
            for agent in self.agents:
                self.losses_list_actor[agent].append(losses_actor[agent].item())
                self.means_list[agent].append(means[agent].mean().detach().squeeze().item())
                self.stds_list[agent].append(stds[agent].mean().detach().squeeze().item())
                self.state_means[agent].append(np.mean(state_stats[agent].mean()))
                self.state_stds[agent].append(np.mean(state_stats[agent].std()))
            self.loss_list_critic.append(loss_critic.item())
            # self.list_state_mean_1.append(state_stat_mean[0])
            # self.list_state_mean_2.append(state_stat_mean[1])
            # self.list_state_std_1.append(state_stat_std[0])
            # self.list_state_std_2.append(state_stat_std[1])

            if iteration % self.plot_per == 0:
                # AX 1
                # self.ax_1.cla()
                # input_values_np = observations_tensor.squeeze().numpy()
                # x = input_values_np[:, 0]
                # y = input_values_np[:, 1]
                #
                # actor_output_tensor_np = actor_output_tensor.detach().squeeze().numpy()
                # self.ax_1.scatter(x, y, actor_output_tensor_np[:, 0], marker='.', alpha=0.09, label='action 1')
                # self.ax_1.scatter(x, y, actor_output_tensor_np[:, 1], marker='x', alpha=0.09, label='action 2')
                # # critic_output_tensor_np = critic_output_tensor.detach().squeeze().numpy()
                # # ax_1.scatter(x, y, critic_output_tensor_np, marker='.', alpha=0.1, label='critic values')
                # self.ax_1.set_title('Outputs of NN')
                # self.ax_1.legend()

                # AX 2
                self.ax_2.cla()
                for agent in self.agents:
                    self.ax_2.plot(self.means_list[agent], label=f'mean ({agent})')
                    self.ax_2.plot(self.stds_list[agent], label=f'std ({agent})', linestyle='--')
                self.ax_2.set_title('Mean & STD')
                self.ax_2.legend()

                # AX 3
                self.ax_3.cla()
                for agent in self.agents:
                    self.ax_3.plot(self.losses_list_actor[agent], label=f'{agent}')
                self.ax_3.plot(self.loss_list_critic, label='critic')
                self.ax_3.set_title('Loss')
                self.ax_3.legend()

                # AX 4
                self.ax_4.cla()
                self.ax_4.plot(self.total_scores, label='scores')
                self.ax_4.plot(self.total_avg_scores, label='avg scores')
                self.ax_4.set_title('Scores')
                self.ax_4.legend()

                # AX 5
                self.ax_5.cla()
                for agent in self.agents:
                    self.ax_5.plot(self.state_means[agent], label=f'mean ({agent})')
                    self.ax_5.plot(self.state_stds[agent], label=f'std ({agent})', linestyle='--')
                self.ax_5.set_title('State stat')
                self.ax_5.legend()

                plt.pause(0.05)

    def plot_close(self):
        if self.plot_life:
            plt.close()

    def neptune_init(self, params):
        if self.plot_neptune:
            self.run = neptune.init(project='1919ars/MA-implementations',
                                    tags=self.tags,
                                    name=f'{self.name}')

            self.run['parameters'] = params
            self.neptune_initiated = True

    def neptune_plot(self, update_dict: dict):
        if self.plot_neptune:

            if not self.neptune_initiated:
                raise RuntimeError('~[ERROR]: Initiate NEPTUNE!')

            for k, v in update_dict.items():
                self.run[k].log(v)

    def neptune_close(self):
        if self.plot_neptune and self.neptune_initiated:
            self.run.stop()







