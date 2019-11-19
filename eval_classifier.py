import argparse
import pickle
import utils
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix

import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

logging = utils.get_logging()


def save_model(input_data, target_data, classifier_path):
    model = utils.get_model()
    class_names = np.unique(target_data)
    model.fit(input_data, target_data)
    with open(classifier_path, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    logging.info('Saved classifier model to file "%s"' % classifier_path)


def main(classifier_data):
    classifier_path = classifier_data.model_path
    if os.path.exists(classifier_path):
        os.remove(classifier_path)
    input_data = classifier_data.get_input_data()
    target_data = classifier_data.get_target_data()
    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(input_data, target_data,
                                                                                    test_size=utils.validation_size,
                                                                                    random_state=utils.seed)
    # evaluate each model in turn
    results = []
    names = []
    for name, model in utils.get_models():
        kfold = model_selection.KFold(n_splits=utils.kfold, random_state=utils.seed)
        cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=utils.scoring, verbose=True)

        predicted = model_selection.cross_val_predict(model, x_train, y_train)
        class_names = np.unique(target_data)
        cm = confusion_matrix(y_train, predicted, labels=class_names)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(name)
        plt.show()
        results.append(cv_results)
        names.append(name)
        msg = "%s: Mean : %f, STD : %f, Max : %f, Min : %f" % (name, cv_results.mean(), cv_results.std(), max(cv_results),
                                                               min(cv_results))
        logging.info(msg)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    save_model(input_data, target_data, classifier_path)
    return True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--classifier-path', type=str, action='store', dest='classifier_path',
                        help='Path to output trained classifier model')
    parser.add_argument('--input-data', type=str, action='store', dest='input_data',
                        help='Path to output trained classifier model')
    parser.add_argument('--target-data', type=str, action='store', dest='target_data',
                        help='Path to output trained classifier model')
    args = parser.parse_args()
    main(classifier_data=utils.ClassifierData(model_path=args.classifier_path, input_path=args.input_data,
                                              target_path=args.target_data))
