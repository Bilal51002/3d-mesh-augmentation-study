import os
import torch

from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer

FICHIER_LOG_ECHECS_TEST = "batchs_test_ignores.log"


def run_test_or_val(phase):
    opt = TestOptions().parse()
    opt.serial_batches = True
    opt.phase = phase
    opt.number_augmentations = 1

    if opt.phase == "val" and opt.dataset_mode == "segmentation" and not os.path.isdir("datasets/human_seg/val"):
        return "No Data", 0.0

    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)

    writer.reset_counter()

    nb_ok = 0
    nb_ignores = 0
    total_loss = 0.0
    nb_batchs_avec_loss = 0

    for i, data in enumerate(dataset):
        try:
            model.set_input(data)
            with torch.no_grad():
                out = model.forward()
                loss = model.criterion(out, model.labels)
                total_loss += loss.item()
                nb_batchs_avec_loss += 1
                predictions = out.data.max(1)[1]
                ncorrect = model.get_accuracy(predictions=predictions, labels=model.labels)
                nexamples = len(model.labels)
            writer.update_counter(ncorrect, nexamples)
            nb_ok += 1
        except Exception as e:
            nb_ignores += 1
            with open(FICHIER_LOG_ECHECS_TEST, "a") as f:
                f.write(f"Phase {phase}, batch {i}: {e}\n")
            continue

    avg_loss = total_loss / nb_batchs_avec_loss if nb_batchs_avec_loss > 0 else 0.0
    print(f"  [Test/{phase}] Batchs OK: {nb_ok} | Batchs ignores: {nb_ignores} | Test Loss: {avg_loss:.5f}")
    return writer.acc, avg_loss


def run_test():
    accuracy, avg_loss = run_test_or_val("test")
    return accuracy, avg_loss


if __name__ == "__main__":
    print("Running Test")
    accuracy, avg_loss = run_test()
    print("Test accuracy: {:.5} % | Test loss: {:.5f}".format(accuracy * 100, avg_loss))
