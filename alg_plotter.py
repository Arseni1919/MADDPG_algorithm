from alg_constrants_amd_packages import *
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
    def __init__(self, plot_life=True, plot_neptune=False):

        self.plot_life = plot_life
        self.plot_neptune = plot_neptune
        self.fig, self.actor_losses, self.critic_losses, self.ax, self.agents_list = {}, {}, {}, {}, {}

        self.neptune_init()
        self.logging_init()
        self.info("ALGPlotter instance created.")

    def plots_set(self, env_module):
        if self.plot_life:
            self.agents_list = env_module.get_agent_list()
            self.fig, self.ax = plt.subplots(nrows=2, ncols=len(self.agents_list), figsize=(12, 6))
            # self.ax = self.fig.get_axes()
            self.actor_losses = {agent: [] for agent in self.agents_list}
            self.critic_losses = {agent: [] for agent in self.agents_list}

    def plots_update_data(self, data_dict, dict_type):
        if self.plot_life:
            for agent_name, value in data_dict.items():
                if dict_type == 'actor':
                    self.actor_losses[agent_name].append(value)
                elif dict_type == 'critic':
                    self.critic_losses[agent_name].append(value)
                else:
                    self.error('Actor type is not correct.')

    def plots_online(self):
        # plot live:
        if self.plot_life:
            def plot_graph(ax, indx_r, indx_c, list_of_values, label, color='b'):
                ax[indx_r, indx_c].cla()
                ax[indx_r, indx_c].plot(list(range(len(list_of_values))), list_of_values, c=color)  # , edgecolor='b')
                ax[indx_r, indx_c].set_title(f'Plot: {label}')
                ax[indx_r, indx_c].set_xlabel('iters')
                ax[indx_r, indx_c].set_ylabel(f'{label}')

            # graphs
            for agent_indx, agent in enumerate(self.agents_list):
                plot_graph(self.ax, 0, agent_indx,  self.actor_losses[agent], f'{agent}_loss')
                plot_graph(self.ax, 1, agent_indx, self.critic_losses[agent], f'{agent}_loss')

            plt.pause(0.05)

    def plot_summary(self):
        pass

    def neptune_init(self):
        if self.plot_neptune:
            self.run = neptune.init(project='1919ars/PL-implementations',
                                    tags=['MADDPG'],
                                    name=f'MADDPG_{time.asctime()}',
                                    source_files=['alg_constrants_amd_packages.py'])
            # Neptune.ai Logger
            PARAMS = {
                'GAMMA': GAMMA,
                # 'LR': LR,
                # 'CLIP_GRAD': CLIP_GRAD,
                # 'MAX_STEPS': MAX_STEPS,
            }
            self.run['parameters'] = PARAMS
        else:
            self.run = {}

    def neptune_update(self, loss):
        if self.plot_neptune:
            self.run['acc_loss'].log(loss)
            self.run['acc_loss_log'].log(f'{loss}')

    @staticmethod
    def logging_init():
        # logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
        # logging.basicConfig(level=logging.DEBUG)
        pass

    def info(self, message, print_info=True, end='\n'):
        # logging.info('So should this')
        if print_info:
            print(colored(f'~[INFO]: {message}', 'green'), end=end)

    def debug(self, message, print_info=True, end='\n'):
        # logging.debug('This message should go to the log file')
        if print_info:
            print(colored(f'~[DEBUG]: {message}', 'cyan'), end=end)

    def warning(self, message, print_info=True, end='\n'):
        # logging.warning('And this, too')
        if print_info:
            print(colored(f'\n~[WARNING]: {message}', 'yellow'), end=end)

    def error(self, message):
        # logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
        raise RuntimeError(f"~[ERROR]: {message}")


plotter = ALGPlotter(
    plot_life=PLOT_LIVE,
    plot_neptune=NEPTUNE
)




