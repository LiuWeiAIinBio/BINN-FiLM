import os
import pandas as pd
from binn import BINN
from trainer import BINNTrainer
import config
from util import set_seed


set_seed(6)
train = True
evaluate = False


# initialize BINN and BINNTrainer
binn_model = BINN(data_matrix=config.data_matrix,
                  mapping=config.mapping,
                  pathways=config.pathways,
                  activation=config.activation,
                  n_layers=config.n_layers,
                  n_outputs=config.n_outputs,
                  dropout=config.dropout)

binntrainer = BINNTrainer(binn_model=binn_model,
                          save_dir=config.save_dir,
                          device=config.device)


if train:
    binntrainer.train(X_train=config.X_train,
                      X_val=config.X_val,
                      T_train=config.T_train,
                      T_val=config.T_val,
                      y_train=config.y_train,
                      y_val=config.y_val,
                      BATCH_SIZE=config.BATCH_SIZE,
                      num_epochs=config.num_epochs,
                      learning_rate=config.learning_rate,
                      checkpoint_path=config.checkpoint_path,
                      weight_decay=config.weight_decay)

    # save config information
    config_dict = {"Date": config.timestamp,
                   "dataset": config.dataset_folder_name,
                   "dropout": config.dropout,
                   "weight_decay": config.weight_decay,
                   "fold0_train_acc": "",
                   "fold1_train_acc": "",
                   "fold2_train_acc": "",
                   "fold0_val_acc": "",
                   "fold1_val_acc": "",
                   "fold2_val_acc": "",
                   "BATCH_SIZE": config.BATCH_SIZE,
                   "num_epochs": config.num_epochs,
                   "learning_rate": config.learning_rate,
                   "activation": config.activation,
                   "device": config.device,
                   "feature_data_dir": config.feature_data_dir,
                   "label_data_dir": config.label_data_dir}

    config_dir = os.path.join(config.checkpoint_path, "..", "config_log.txt")
    max_key_len = max(len(k) for k in config_dict)

    with open(config_dir, "w") as f:
        for key, value in config_dict.items():
            f.write(f"{key:<{max_key_len + 2}} {value}\n")

    run_log_path = os.path.join(config.base_dir, "..", "run_log.csv")
    if os.path.exists(path=run_log_path):
        existing_df = pd.read_csv(run_log_path)
        combined_df = pd.concat([existing_df, pd.DataFrame(config_dict, index=[0])], ignore_index=True)
        combined_df.to_csv(run_log_path, index=False)
    else:
        pd.DataFrame(config_dict, index=[0]).to_csv(run_log_path, index=False)


# evaluate generation of the model
if evaluate:
    checkpoint_path = "D:\\Desktop\\binn_4_2\\output\\20250809_200858\\checkpoint\\checkpoint_epoch_9.pt"  # not fixed
    evaluate_save_dir = os.path.join(checkpoint_path, "..", "..", "logs")

    binntrainer.evaluate(evaluate_feature_dir=config.evaluate_feature_dir, evaluate_label_dir=config.evaluate_label_dir,
                         BATCH_SIZE=config.BATCH_SIZE, evaluate_save_dir=evaluate_save_dir, path_checkpoint=checkpoint_path)
