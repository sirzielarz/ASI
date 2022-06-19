from insurance.model import read_and_train
from insurance.evaluation import evaluate
from insurance.drift import drift

if __name__ == '__main__':
    read_and_train.prepare_data()
    read_and_train.train()
    for s in range(10):
        evaluate.run("model/model.pkl", s)
    drift.detect()
    for s in range(10):
        evaluate.run("model/model.pkl", s)
    drift.detect()
