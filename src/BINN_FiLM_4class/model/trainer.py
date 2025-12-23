import copy
import os
import numpy as np
import pandas as pd
import torch
import seaborn
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import BINNDataset
from util import compute_loss, plot_logits, focal_loss, compute_loss1
import config


class BINNTrainer:
    def __init__(self, binn_model, save_dir: str, device: str):
        self.binn_model = binn_model
        self.device = device
        self.binn_model.to(self.device)
        self.save_dir = save_dir
        self.logger = BINNLogger(self.save_dir)
        self.init_state = copy.deepcopy(binn_model.state_dict())  # 用 deepcopy 保存一份完全独立的初始参数
        os.makedirs(self.save_dir, exist_ok=True)


    def train(self,
              X_train: pd.DataFrame,
              X_val: pd.DataFrame,
              T_train: pd.DataFrame,
              T_val: pd.DataFrame,
              y_train: pd.DataFrame,
              y_val: pd.DataFrame,
              BATCH_SIZE: int,
              num_epochs: int,
              learning_rate: float,
              checkpoint_path: str = None,
              weight_decay: float = 0.01):

        for fold in range(5):
            self.binn_model.load_state_dict(copy.deepcopy(self.init_state))  # 每个 fold 前将模型参数重置为初始参数
            self.binn_model.to(self.device)

            train_data = BINNDataset(X_train[fold], T_train[fold], y_train[fold])
            val_data = BINNDataset(X_val[fold], T_val[fold], y_val[fold])
            train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

            optimizer = torch.optim.Adam(self.binn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            all_avg_train_loss = []
            all_avg_train_accuracy = []
            all_avg_val_loss = []
            all_avg_val_accuracy = []

            # create SummaryWriter
            writer = SummaryWriter(comment='BINN_train')

            for epoch in range(num_epochs):
                self.binn_model.train()
                train_loss, train_accuracy = 0.0, 0.0

                final_out_logits_val = []
                logits_labels_val = []

                for i, data in enumerate(train_loader):
                    inputs, task_ids, labels = data
                    inputs = inputs.to(self.device)
                    task_ids = task_ids.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.binn_model(inputs, task_ids)

                    # index_0暂存复原标签0/1/2
                    # index_0 = torch.zeros(outputs.shape[0], dtype=torch.long)
                    # mask0 = ((task_ids == 0) & (labels == 0)) | ((task_ids == 1) & (labels == 0))
                    # mask1 = ((task_ids == 0) & (labels == 1)) | ((task_ids == 2) & (labels == 0))
                    # mask2 = ((task_ids == 1) & (labels == 1)) | ((task_ids == 2) & (labels == 1))
                    # index_0[mask0] = 0
                    # index_0[mask1] = 1
                    # index_0[mask2] = 2

                    # backward propagation
                    optimizer.zero_grad()
                    # loss = focal_loss(outputs, labels, index_0)
                    loss = compute_loss(outputs, labels)
                    loss.backward()

                    # update weights
                    optimizer.step()

                    train_loss += loss.item()
                    train_accuracy += (torch.argmax(outputs, dim=1) == labels).float().mean().item()

                # record the parameters of each network layer and the gradient of the parameters for each epoch
                for layer_name, layer_param in self.binn_model.named_parameters():
                    writer.add_histogram(layer_name + '_grad', layer_param.grad, epoch)
                    writer.add_histogram(layer_name + '_data', layer_param.data, epoch)

                avg_train_loss = train_loss / len(train_loader)
                avg_train_accuracy = train_accuracy / len(train_loader)

                # transmit train metrics of one epoch into logger
                train_metrics = {f"epoch_{epoch}_avg_train_loss": avg_train_loss,
                                 f"epoch_{epoch}_avg_train_accuracy": avg_train_accuracy}
                self.logger.log(phase="train", metrics=train_metrics)
                # plot
                all_avg_train_loss.append(avg_train_loss)
                all_avg_train_accuracy.append(avg_train_accuracy)
                # save loss and accuracy to event file for train
                writer.add_scalar("train_loss", avg_train_loss, epoch)
                writer.add_scalar("train_accuracy", avg_train_accuracy, epoch)

                # val
                self.binn_model.eval()
                val_loss, val_accuracy = 0.0, 0.0

                with torch.no_grad():
                    for i, data in enumerate(val_loader):
                        inputs, task_ids, labels = data
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        # # trade off 2 tasks
                        # outputs_0 = self.binn_model(inputs, torch.zeros(BATCH_SIZE))
                        # outputs_1 = self.binn_model(inputs, torch.ones(BATCH_SIZE))
                        # outputs_cat = torch.cat([outputs_0, outputs_1], dim=1)
                        # index = torch.argmax(outputs_cat, dim=1)

                        # # 建立用于计算准确率的index_1
                        # index_1 = index.clone()
                        # mask_02 = ((index == 0) | (index == 2))
                        # mask_3 = (index == 3)
                        # index_1[mask_02] = 0
                        # index_1[mask_3] = 1
                        #
                        # # 选出用于计算loss的outputs
                        # mask_0 = ((index == 0) | (index == 1))
                        # outputs = torch.where(mask_0.unsqueeze(-1), outputs_0, outputs_1)

                        # trade off 6 tasks
                        outputs_0 = self.binn_model(inputs, torch.zeros(inputs.shape[0], device=self.device))
                        outputs_1 = self.binn_model(inputs, torch.ones(inputs.shape[0], device=self.device))
                        outputs_2 = self.binn_model(inputs, torch.full((inputs.shape[0],), 2, dtype=torch.float32, device=self.device))
                        outputs_3 = self.binn_model(inputs, torch.full((inputs.shape[0],), 3, dtype=torch.float32, device=self.device))
                        outputs_4 = self.binn_model(inputs, torch.full((inputs.shape[0],), 4, dtype=torch.float32, device=self.device))
                        outputs_5 = self.binn_model(inputs, torch.full((inputs.shape[0],), 5, dtype=torch.float32, device=self.device))

                        prob0 = torch.softmax(outputs_0, dim=1)
                        prob1 = torch.softmax(outputs_1, dim=1)
                        prob2 = torch.softmax(outputs_2, dim=1)
                        prob3 = torch.softmax(outputs_3, dim=1)
                        prob4 = torch.softmax(outputs_4, dim=1)
                        prob5 = torch.softmax(outputs_5, dim=1)
                        prob_cat = torch.cat([prob0, prob1, prob2, prob3, prob4, prob5], dim=1)

                        # 复原标签：四类分类
                        index_0 = torch.full((inputs.shape[0],), -1, dtype=torch.long, device=self.device)
                        mask0 = ((task_ids == 0) & (labels == 0)) | ((task_ids == 1) & (labels == 0)) | ((task_ids == 2) & (labels == 0))
                        mask1 = ((task_ids == 0) & (labels == 1)) | ((task_ids == 3) & (labels == 0)) | ((task_ids == 4) & (labels == 0))
                        mask2 = ((task_ids == 1) & (labels == 1)) | ((task_ids == 3) & (labels == 1)) | ((task_ids == 5) & (labels == 0))
                        mask3 = ((task_ids == 2) & (labels == 1)) | ((task_ids == 4) & (labels == 1)) | ((task_ids == 5) & (labels == 1))
                        index_0[mask0] = 0
                        index_0[mask1] = 1
                        index_0[mask2] = 2
                        index_0[mask3] = 3

                        # 四类分类，index_1暂存预测结果0/1/2/3
                        # 算法1
                        index_1 = torch.full((inputs.shape[0],), -2, dtype=torch.long, device=self.device)
                        mask_0 = (torch.argmax(prob0, dim=1) == 0) & (torch.argmax(prob1, dim=1) == 0) & (torch.argmax(prob2, dim=1) == 0)
                        mask_1 = (torch.argmax(prob0, dim=1) == 1) & (torch.argmax(prob3, dim=1) == 0) & (torch.argmax(prob4, dim=1) == 0)
                        mask_2 = (torch.argmax(prob1, dim=1) == 1) & (torch.argmax(prob3, dim=1) == 1) & (torch.argmax(prob5, dim=1) == 0)
                        mask_3 = (torch.argmax(prob2, dim=1) == 1) & (torch.argmax(prob4, dim=1) == 1) & (torch.argmax(prob5, dim=1) == 1)
                        index_1[mask_0] = 0
                        index_1[mask_1] = 1
                        index_1[mask_2] = 2
                        index_1[mask_3] = 3
                        # 算法2
                        # index_1 = torch.full((inputs.shape[0],), -2, dtype=torch.long, device=self.device)
                        # mask_0 = ((torch.argmax(prob0, dim=1) == 0) & (torch.argmax(prob1, dim=1) == 0)) | (
                        #             (torch.argmax(prob0, dim=1) == 0) & (torch.argmax(prob2, dim=1) == 0)) | (
                        #                      (torch.argmax(prob1, dim=1) == 0) & (torch.argmax(prob2, dim=1) == 0))
                        # mask_1 = ((torch.argmax(prob0, dim=1) == 1) & (torch.argmax(prob3, dim=1) == 0)) | (
                        #             (torch.argmax(prob0, dim=1) == 1) & (torch.argmax(prob4, dim=1) == 0)) | (
                        #                      (torch.argmax(prob3, dim=1) == 0) & (torch.argmax(prob4, dim=1) == 0))
                        # mask_2 = ((torch.argmax(prob1, dim=1) == 1) & (torch.argmax(prob3, dim=1) == 1)) | (
                        #             (torch.argmax(prob1, dim=1) == 1) & (torch.argmax(prob5, dim=1) == 0)) | (
                        #                      (torch.argmax(prob1, dim=1) == 1) & (torch.argmax(prob5, dim=1) == 0))
                        # mask_3 = ((torch.argmax(prob2, dim=1) == 1) & (torch.argmax(prob4, dim=1) == 1)) | (
                        #             (torch.argmax(prob2, dim=1) == 1) & (torch.argmax(prob5, dim=1) == 1)) | (
                        #                      (torch.argmax(prob4, dim=1) == 1) & (torch.argmax(prob5, dim=1) == 1))
                        # index_1[mask_0] = 0
                        # index_1[mask_1] = 1
                        # index_1[mask_2] = 2
                        # index_1[mask_3] = 3

                        # 选出outputs计算loss
                        outputs_cat = torch.cat([outputs_0, outputs_1, outputs_2, outputs_3, outputs_4, outputs_5], dim=1)
                        outputs = torch.zeros(outputs_cat.shape[0], 4)
                        outputs[mask_0] = outputs_cat[mask_0][:, [0, 1, 3, 5]]
                        outputs[mask_1] = outputs_cat[mask_1][:, [0, 1, 7, 9]]
                        outputs[mask_2] = outputs_cat[mask_2][:, [2, 6, 3, 11]]
                        outputs[mask_3] = outputs_cat[mask_3][:, [4, 8, 10, 5]]

                        # # 对齐outputs创建prob，仅查看012
                        prob = torch.zeros(prob_cat.shape[0], 3, device=self.device)
                        prob[mask_0] = prob_cat[mask_0][:, [0, 1, 3]]
                        prob[mask_1] = prob_cat[mask_1][:, [0, 1, 7]]
                        prob[mask_2] = prob_cat[mask_2][:, [2, 6, 3]]
                        prob_temp = prob[~mask_3]
                        index_0_temp = index_0[~mask_3]
                        print(epoch, "\n", prob_cat, "\n", index_0, "\n", index_1, "\n", prob)

                        loss = compute_loss(outputs, index_0)

                        val_loss += loss.item()
                        val_accuracy += (index_0 == index_1).float().mean().item()

                    #     if epoch % 4999 == 0:
                    #         final_out_logits_val.append(prob_temp)
                    #         logits_labels_val.append(index_0_temp)
                    #
                    # if final_out_logits_val:
                    #     plot_logits(torch.cat(final_out_logits_val, dim=0), labels=torch.cat(logits_labels_val, dim=0),
                    #                 description=f'val_fold{fold}', save_dir=config.save_dir)

                avg_val_loss = val_loss / len(val_loader)
                avg_val_accuracy = val_accuracy / len(val_loader)

                # transmit val metrics of one epoch into logger
                val_metrics = {f"epoch_{epoch}_avg_val_loss": avg_val_loss,
                               f"epoch_{epoch}_avg_val_accuracy": avg_val_accuracy}
                self.logger.log(phase="val", metrics=val_metrics)
                # plot
                all_avg_val_loss.append(avg_val_loss)
                all_avg_val_accuracy.append(avg_val_accuracy)
                # save loss and accuracy to event file for val
                writer.add_scalar("val_loss", avg_val_loss, epoch)
                writer.add_scalar("val_accuracy", avg_val_accuracy, epoch)

                # for name, param in self.binn_model.named_parameters():
                #     print(f"Parameter name: {name}")
                #     print(f"Parameter value: {param.data}")

                if checkpoint_path:
                    os.makedirs(checkpoint_path, exist_ok=True)
                    if (epoch + 1) % 2 == 0:  # checkpoint interval: 50
                        checkpoint = {"model_state_dict": self.binn_model.state_dict(),
                                      "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch}
                        path_checkpoint = f"{checkpoint_path}/fold{fold}_checkpoint_epoch_{epoch}.pt"
                        torch.save(checkpoint, path_checkpoint)

            self.logger.save_logs(fold)  # save train and val metrics of all epochs as CSV file

            # use fake data to create model's computational graph
            # fake_data = torch.randn((5, 233))
            # writer.add_graph(model=self.binn_model, input_to_model=fake_data, verbose=False)

            writer.close()


            # create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            fig.suptitle('BINN_FiLM model train log', fontsize=16)

            # 1: loss curve
            ax1.plot(range(num_epochs), all_avg_train_loss, 'r-', label='Train loss')
            ax1.plot(range(num_epochs), all_avg_val_loss, 'b-', label='Val loss')
            ax1.set_title('Train and val set loss curve')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)

            # 2: accuracy curve
            ax2.plot(range(num_epochs), all_avg_train_accuracy, 'r-', label=f'Train accuracy (avg: {sum(all_avg_train_accuracy) / num_epochs})')
            ax2.plot(range(num_epochs), all_avg_val_accuracy, 'b-', label=f'Val accuracy (avg: {sum(all_avg_val_accuracy) / num_epochs})')
            ax2.set_title('Train and val set accuracy curve')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)

            # adjust layout
            plt.tight_layout()

            # save figure
            save_path = os.path.join(self.save_dir, f'fold{fold}_train_val_log_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')


    def evaluate(self, evaluate_feature_dir, evaluate_label_dir, BATCH_SIZE: int, evaluate_save_dir: str, path_checkpoint: str):
        evaluate_feature_df = pd.read_csv(evaluate_feature_dir)
        evaluate_label_df = pd.read_csv(evaluate_label_dir)
        evaluate_data = BINNDataset(evaluate_feature_df, evaluate_label_df)
        evaluate_loader = DataLoader(dataset=evaluate_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        checkpoint = torch.load(path_checkpoint, weights_only=True)
        self.binn_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.binn_model.eval()

        all_labels, all_preds, all_probs = [], [], []
        evaluate_loss, evaluate_accuracy = 0.0, 0.0

        with torch.no_grad():
            for i, data in enumerate(evaluate_loader):
                inputs, task_ids, labels = data
                inputs = inputs.to(self.device)
                task_ids = task_ids.to(self.device)
                labels = labels.to(self.device)
                outputs = self.binn_model(inputs, task_ids)

                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                loss = compute_loss(outputs, labels)
                evaluate_loss += loss.item()
                evaluate_accuracy += (torch.argmax(outputs, dim=1) == labels).float().mean().item()

                all_labels.append(labels.numpy())
                all_preds.append(preds.numpy())
                all_probs.append(probs.numpy())

        # concatenate all batches
        y_true = np.concatenate(all_labels)  # Concatenate all_labels into a one-dimensional NumPy array containing all true labels of the evaluation set
        y_pred = np.concatenate(all_preds)
        y_score = np.concatenate(all_probs)
        avg_evaluate_loss = evaluate_loss / len(evaluate_loader)
        avg_evaluate_accuracy = evaluate_accuracy / len(evaluate_loader)

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        class_names = ["cont", "CO19-ARDS", "Seps-ARDS"]  # 0=cont，1=CO19-ARDS，2=Seps-ARDS

        fig, ax = plt.subplots(figsize=(6, 5))
        seaborn.heatmap(cm_percent, annot=True, fmt=".2%", cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                        cbar=True, square=True, linewidths=.5, ax=ax)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j + 0.5, i + 0.5, f"\n{cm[i, j]}", ha='center', va='center', color='black', fontsize=10)  # mark the original quantities in each cell

        ax.set_xlabel('Pred_label')
        ax.set_ylabel('True_label')
        ax.set_title('BINN_FiLM 3-class confusion matrix')
        cm_path = os.path.join(evaluate_save_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Evaluate Report
        report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        report_dict.update({"avg_evaluate_loss": avg_evaluate_loss, "avg_evaluate_accuracy": avg_evaluate_accuracy})
        report_df = pd.DataFrame(report_dict).transpose()
        report_csv = os.path.join(evaluate_save_dir, "classification_report.csv")
        report_df.to_csv(report_csv, index=True)
        print("Evaluate Report：\n", report_df)

        # ROC
        y_bin = label_binarize(y_true, classes=[0, 1, 2])
        plt.figure(figsize=(7, 6))
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", lw=2)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("BINN_FiLM 3-class ROC curve")
        plt.legend(loc="lower right")
        roc_path = os.path.join(evaluate_save_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()



class BINNLogger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.logs = {"train": [], "val": [], "evaluate": []}

    def log(self, phase, metrics):
        """
        Log metrics for a specific phase (train/val/evaluate).
        Args:
            phase (str): Phase name ('train' or 'val' or 'evaluate').
            metrics (dict): Dictionary containing metric names and values.
        """
        self.logs[phase].append(metrics)

    def save_logs(self, fold):
        """
        Save logs to disk as a CSV file.
        """
        for phase, log_data in self.logs.items():
            if log_data:
                df = pd.DataFrame(log_data)
                df.to_csv(f"{self.save_dir}/fold{fold}_{phase}_logs.csv", index=False)
