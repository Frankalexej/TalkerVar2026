from dataclasses import replace
from src.four_feature_config import FourFeatureConfig
CONFIG = FourFeatureConfig(seeds=(42,),epochs=1,pretrain_epochs=1,output_dir='outputs/four_feature_smoke',
                           kmeans_n_init=1,gmm_n_init=1,gmm_max_iter=30)
CONFIG.baseline_training = replace(CONFIG.baseline_training,epochs=1,pretrain_epochs=1)
