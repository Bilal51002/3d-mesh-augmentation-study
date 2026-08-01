import os

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
        return "No Data"

    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)

    writer.reset_counter()

    nb_ok = 0
    nb_ignores = 0
    for i, data in enumerate(dataset):
        try:
            model.set_input(data)
            ncorrect, nexamples = model.test()
            writer.update_counter(ncorrect, nexamples)
            nb_ok += 1
        except Exception as e:
            nb_ignores += 1
            with open(FICHIER_LOG_ECHECS_TEST, "a") as f:
                f.write(f"Phase {phase}, batch {i}: {e}\n")
            continue

    print(f"  [Test/{phase}] Batchs OK: {nb_ok} | Batchs ignores: {nb_ignores}")
    return writer.acc


def run_test():
    accuracy = run_test_or_val("test")
    return accuracy


if __name__ == "__main__":
    print("Running Test")
    accuracy = run_test()
    print("Test accuracy: {:.5} %".format(accuracy * 100))
