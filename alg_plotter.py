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
    def __init__(self):
        if PLOT_LIVE:
            self.fig, _ = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
            self.actor_losses = []
            self.critic_losses_1 = []
            self.critic_losses_2 = []

        self.neptune_init()
        self.logging_init()
        self.info("ALGPlotter instance created.")

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

    def neptune_init(self):
        if NEPTUNE:
            self.run = neptune.init(project='1919ars/PL-implementations',
                                    tags=['MADDPG'],
                                    name=f'MADDPG_{time.asctime()}',
                                    source_files=['alg_constrants_amd_packages.py'])
            # Neptune.ai Logger
            PARAMS = {
                'GAMMA': GAMMA,
                # 'LR': LR,
                # 'CLIP_GRAD': CLIP_GRAD,
                'MAX_STEPS': MAX_STEPS,
            }
            self.run['parameters'] = PARAMS
        else:
            self.run = {}

    def neptune_update(self, loss):
        if NEPTUNE:
            self.run['acc_loss'].log(loss)
            self.run['acc_loss_log'].log(f'{loss}')

    @staticmethod
    def logging_init():
        # logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
        # logging.basicConfig(level=logging.DEBUG)
        pass

    def info(self, message, print_info=True):
        # logging.info('So should this')
        if print_info:
            print(colored(f'~[INFO]: {message}', 'green'))

    def debug(self, message, print_info=True):
        # logging.debug('This message should go to the log file')
        if print_info:
            print(colored(f'~[DEBUG]: {message}', 'cyan'))

    def warning(self, message, print_info=True):
        # logging.warning('And this, too')
        if print_info:
            print(colored(f'~[WARNING]: {message}', 'yellow'))

    def error(self, message):
        # logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
        raise RuntimeError(f"~[ERROR]: {message}")






plotter = ALGPlotter()




