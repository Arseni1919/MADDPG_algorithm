from alg_env_module import *
from alg_plotter import plotter

plotter.info("Perfect!!!! Vey good:)")
plotter.debug("Perfect!!!! Vey good:)")
plotter.warning("Perfect!!!! Vey good:)")
plotter.error("Perfect!!!! Vey good:)")
# logging.basicConfig(level=logging.DEBUG)
# logging.debug('This message should go to the log file')
# logging.info('So should this')
# logging.warning('And this, too')
# logging.error('And non-ASCII stuff, too, like Øresund and Malmö')


env = ALGEnv_Module(ENV)

for i in range(10):
    print(i)
    print(env.run_episode(render=True))

print('waaa')

