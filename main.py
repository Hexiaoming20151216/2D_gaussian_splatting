import os.path

from gaussian_splatting_2D_optimizer import GaussianSplatting2dOptimizer


gs2dopt = GaussianSplatting2dOptimizer(os.path.dirname(os.path.abspath(__file__)) + '/config.yml')
gs2dopt.load_image()
gs2dopt.create_dir()
gs2dopt.run_opt_loop()
gs2dopt.save_result()
