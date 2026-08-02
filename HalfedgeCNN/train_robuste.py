import os
import time
import json
from os.path import join
from options.train_options import TrainOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from test_robuste import run_test
from validate import run_validation

FICHIER_LOG_ECHECS = "batchs_ignores.log"


def train_one_epoch():
    epoch_start_time = time.time()
    epoch_iter = 0
    nb_batchs_ok = 0
    nb_batchs_ignores = 0
    nb_correct_total = 0
    nb_echantillons_total = 0
    for i, data in enumerate(dataset):
        try:
            model.set_input(data)
            model.optimize_parameters()
            with torch.no_grad():
                out = model.net(model.half_edge_features, model.meshes)
                predictions = out.data.max(1)[1]
                nb_correct_total += (predictions == model.labels).sum().item()
                nb_echantillons_total += len(model.labels)
            nb_batchs_ok += 1
        except Exception as e:
            nb_batchs_ignores += 1
            with open(FICHIER_LOG_ECHECS, "a") as f:
                f.write(f"Epoch {epoch}, batch {i}: {e}\n")
            continue
        epoch_iter += opt.batch_size
        writer.plot_loss(model.loss, epoch, epoch_iter, len(dataset))
    training_time = time.time() - epoch_start_time
    train_accuracy = nb_correct_total / nb_echantillons_total if nb_echantillons_total > 0 else 0.0
    print(f"  [Epoch {epoch}] Batchs OK: {nb_batchs_ok} | Batchs ignores: {nb_batchs_ignores} | Train Acc: {train_accuracy*100:.2f}%")
    return training_time, train_accuracy


def test():
    start_time = time.time()
    accuracy, avg_loss = run_test()
    return accuracy, avg_loss, time.time() - start_time


def validate():
    start_time = time.time()
    accuracy = run_validation()
    return accuracy, time.time() - start_time


def init_writer():
    logging_header = ("Epoch", "Training Loss", "Train Accuracy", "Test Loss", "Test Accuracy", "Training Time", "Test Time", "Total Time", "              Learn Rate", "Model saved")
    writer = Writer(opt, logging_header)
    return writer


def log_epoch_data():
    learn_rate = model.optimizer.param_groups[0]["lr"]
    data = (epoch, model.loss.item(), train_accuracy, test_loss, test_accuracy, training_time, test_time, total_time, learn_rate, best_model_saved)
    writer.log_epoch_data(data)


if __name__ == "__main__":
    import torch
    train_options = TrainOptions()
    opt = train_options.parse()
    dataset = DataLoader(opt)
    model = create_model(opt)

    writer = init_writer()
    writer.log_options(train_options)
    writer.log_model_description(model)
    writer.log_headline()

    best_accuracy = 0.0
    if opt.continue_train:
        chemin_best_acc = join(opt.checkpoints_dir, opt.name, "best_accuracy.json")
        if os.path.exists(chemin_best_acc):
            with open(chemin_best_acc) as f_acc:
                best_accuracy = json.load(f_acc)["best_accuracy"]
            print(f"Reprise : meilleure accuracy connue = {best_accuracy*100:.2f}%")

    num_epochs = opt.niter + opt.niter_decay
    for epoch in range(opt.epoch_count, num_epochs + 1):
        start_time = time.time()

        training_time, train_accuracy = train_one_epoch()
        model.save_network("latest")

        if epoch % opt.run_test_freq == 0:
            test_accuracy, test_loss, test_time = test()

            best_model_saved = False
            if test_accuracy > best_accuracy:
                best_model_saved = True
                model.save_network("best")
                best_accuracy = test_accuracy
                with open(join(opt.checkpoints_dir, opt.name, "best_accuracy.json"), "w") as f_acc:
                    json.dump({"best_accuracy": best_accuracy, "epoch": epoch}, f_acc)

            total_time = time.time() - start_time
            log_epoch_data()
            writer.plot_acc(test_accuracy, epoch)

        model.update_learning_rate()

    writer.close()
    print(f"\nEntrainement termine. Voir {FICHIER_LOG_ECHECS} pour les batchs ignores.")
