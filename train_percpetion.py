"""
@author Feiyang Cai
@email feiyang.cai@vanderbilt.edu
@create date 2020-02-25 09:27:11
@modify date 2020-02-25 09:27:11
@desc main codes to train the perception module
"""

from scripts.perception_trainer import PerceptionTrainer

data_path = "../robust_vae/data/training/"
trainer = PerceptionTrainer(data_path, 350)
trainer.fit()
trainer.save_model()
