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
    def __init__(self, plot_life=True, plot_neptune=False, name='nameless', tags=None):

        if tags is None:
            self.tags = []
        self.plot_life = plot_life
        self.plot_neptune = plot_neptune
        self.fig, self.actor_losses, self.critic_losses, self.ax, self.agents_list = {}, {}, {}, {}, {}
        self.total_reward, self.val_total_rewards = [], []
        self.name = name

        self.run = {}
        self.neptune_initiated = False
        print(colored(f'~[INFO]: "ALGPlotter instance created."', 'green'))

    def plots_online(self):
        if self.plot_life:
            pass

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







