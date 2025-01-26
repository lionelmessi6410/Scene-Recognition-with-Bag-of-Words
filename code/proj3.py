from __future__ import print_function
from random import shuffle
import os
import argparse
import pickle
import sklearn
import numpy

from get_image_paths import get_image_paths
from get_tiny_images import get_tiny_images
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from visualize import visualize

from nearest_neighbor_classify import nearest_neighbor_classify
from svm_classify import svm_classify
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold


# Step 0: Set up parameters, category list, and image paths.

#For this project, you will need to report performance for three
#combinations of features / classifiers. It is suggested you code them in
#this order, as well:
# 1) Tiny image features and nearest neighbor classifier
# 2) Bag of sift features and nearest neighbor classifier
# 3) Bag of sift features and linear SVM classifier
#The starter code is initialized to 'placeholder' just so that the starter
#code does not crash when run unmodified and you can get a preview of how
#results are presented.

parser = argparse.ArgumentParser()
parser.add_argument('--feature', help='feature', type=str, default='dumy_feature')
parser.add_argument('--classifier', help='classifier', type=str, default='dumy_classifier')
args = parser.parse_args()


DATA_PATH = '../data/'
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TEST_PATH = os.path.join(DATA_PATH, 'test')

#This is the list of categories / directories to use. The categories are
#somewhat sorted by similarity so that the confusion matrix looks more
#structured (indoor and then urban and then rural).

CATEGORIES = [d for d in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, d))]
CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

ABBR_CATEGORIES = ['Agri', 'Airp', 'Base', 'Bech', 'Bldg', 'Chap', 'DRes',
                   'Frst', 'Frwy', 'Golf', 'Hbr', 'Intr', 'MRes', 'MblH',
                   'Over', 'Park', 'Rivr', 'Rnwy', 'SRes', 'Stor', 'Tenn']


FEATURE = args.feature
# FEATUR  = 'bag of sift'

CLASSIFIER = args.classifier
# CLASSIFIER = 'support vector machine'

#number of training examples per category to use. Max is 100. For
#simplicity, we assume this is the number of test cases per category, as
#well.

NUM_TRAIN_PER_CAT = 100


def main():
    print("Getting paths and labels for train and test data")
    train_image_paths, train_labels = get_image_paths(TRAIN_PATH, CATEGORIES)
    test_image_paths, test_labels = get_image_paths(TEST_PATH, CATEGORIES)

    # Step 1: Create a per-class split for training and validation
    class_to_images = {category: [] for category in CATEGORIES}
    for path, label in zip(train_image_paths, train_labels):
        class_to_images[label].append(path)
    for category, images in class_to_images.items():
        print(f"{category}:{len(images)}")

    # Ensure each class has at least 80 images (70 for training and 10 for validation)
    for category, images in class_to_images.items():
        print(f"{category}:{len(images)}")
        if len(images) < 80:
            raise ValueError(f"Not enough images in class '{category}' for 70 training and 10 validation split.")

    # Split training data into 70 images for training and 10 for validation per class
    train_split_paths = []
    train_split_labels = []
    val_split_paths = []
    val_split_labels = []

    for category, images in class_to_images.items():
        np.random.seed(42)  # Ensure reproducibility
        np.random.shuffle(images)
        train_split_paths.extend(images[:70])
        train_split_labels.extend([category] * 70)
        val_split_paths.extend(images[70:80])
        val_split_labels.extend([category] * 10)

    # Step 2: Implement k-fold cross-validation on training data
    if FEATURE == 'bag_of_sift':
        if os.path.isfile('vocab.pkl') is False:
            vocab_sizes = [50, 100, 200, 400, 800]
            k = 5  # Number of folds for cross-validation
            print(f"Performing {k}-fold cross-validation to find optimal vocabulary size")
            best_vocab_size = None
            best_accuracy = 0
            accuracies = {vocab_size: [] for vocab_size in vocab_sizes}

            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            for vocab_size in vocab_sizes:
                print(f"Evaluating vocab size: {vocab_size}")
                fold_accuracies = []

                for train_idx, val_idx in kf.split(train_split_paths):
                    # Split train data into training and validation folds
                    fold_train_paths = [train_split_paths[i] for i in train_idx]
                    fold_val_paths = [train_split_paths[i] for i in val_idx]
                    fold_train_labels = [train_split_labels[i] for i in train_idx]
                    fold_val_labels = [train_split_labels[i] for i in val_idx]

                    # Build vocabulary for the current fold
                    vocab = build_vocabulary(fold_train_paths, vocab_size)

                    # Extract features
                    fold_train_feats = get_bags_of_sifts(fold_train_paths, vocab)
                    fold_val_feats = get_bags_of_sifts(fold_val_paths, vocab)

                    # Classify and compute accuracy
                    predicted_labels = svm_classify(fold_train_feats, fold_train_labels, fold_val_feats)
                    fold_accuracy = compute_accuracy(predicted_labels, fold_val_labels)
                    fold_accuracies.append(fold_accuracy)

                # Store mean accuracy for this vocab size
                mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
                accuracies[vocab_size].append(mean_accuracy)
                print(f"Vocab size {vocab_size}: Mean Accuracy = {mean_accuracy:.2f}")

                # Update the best vocabulary size
                if mean_accuracy > best_accuracy:
                    best_vocab_size = vocab_size
                    best_accuracy = mean_accuracy

            # Plot accuracy vs. vocabulary size
            mean_accuracies = [sum(acc) / len(acc) for acc in accuracies.values()]
            plt.plot(vocab_sizes, mean_accuracies, marker='o')
            plt.title('Accuracy vs. Vocabulary Size (k-fold)')
            plt.xlabel('Vocabulary Size')
            plt.ylabel('Accuracy')
            plt.grid()
            plt.savefig('Accuracy_vs_Vocabsize_kfold.png')
            print(f"Optimal vocabulary size: {best_vocab_size}")

            # Build vocabulary with the optimal size
            print(f"Building vocabulary with size: {best_vocab_size}")
            vocab = build_vocabulary(train_split_paths, best_vocab_size)
            with open('vocab.pkl', 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # Extract features for train and test sets
            print("Extracting features for training and testing sets")
            if os.path.isfile('train_image_feats.pkl') is False:
                train_image_feats = get_bags_of_sifts(train_image_paths, vocab)
                with open('train_image_feats.pkl', 'wb') as handle:
                    pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open('train_image_feats.pkl', 'rb') as handle:
                    train_image_feats = pickle.load(handle)
            
            if os.path.isfile('test_image_feats.pkl') is False:
                test_image_feats = get_bags_of_sifts(test_image_paths, vocab)
                with open('test_image_feats.pkl', 'wb') as handle:
                    pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open('test_image_feats.pkl', 'rb') as handle:
                    test_image_feats = pickle.load(handle)

    elif FEATURE == 'tiny_image':
        print("Using tiny images as features")
        train_image_feats = get_tiny_images(train_split_paths)
        test_image_feats = get_tiny_images(test_image_paths)

    elif FEATURE == 'dumy_feature':
        train_image_feats = []
        test_image_feats = []
    else:
        raise NameError('Unknown feature type')

    # Step 3: Classify test images and evaluate accuracy
    print("Classifying test set")
    if CLASSIFIER == 'nearest_neighbor':
        predicted_categories = nearest_neighbor_classify(train_image_feats, train_split_labels, test_image_feats)
    elif CLASSIFIER == 'support_vector_machine':
        predicted_categories = svm_classify(train_image_feats, train_split_labels, test_image_feats)
    else:
        raise NameError('Unknown classifier type')

    # Compute and print overall accuracy
    accuracy = compute_accuracy(predicted_categories, test_labels)
    print(f"Test Accuracy: {accuracy:.2f}")

    # Compute and print class-wise accuracy
    for category in CATEGORIES:
        category_accuracy = compute_accuracy(predicted_categories, test_labels, category)
        print(f"{category}: {category_accuracy:.2f}")

    # Step 4: Build and display confusion matrix
    print("Building confusion matrix")
    test_labels_ids = [CATE2ID[x] for x in test_labels]
    predicted_categories_ids = [CATE2ID[x] for x in predicted_categories]

    build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES)

    # Step 5: Perform t-SNE visualization of SIFT descriptors
    if FEATURE == 'bag_of_sift':
        print("Performing t-SNE visualization")
        _, all_train_descriptors = get_bags_of_sifts(train_image_paths, vocab)
        visualize_tsne(all_train_descriptors, train_labels)


# def main():
#     print("Getting paths and labels for train, validation, and test data")
#     train_image_paths, train_labels = get_image_paths(TRAIN_PATH, CATEGORIES)
#     test_image_paths, test_labels = get_image_paths(TEST_PATH, CATEGORIES)

#     # Step 1: Find the optimal vocabulary size using the validation set
#     if FEATURE == 'bag_of_sift':
#         if os.path.isfile('vocab.pkl') is False:
#             print('No existing visual word vocabulary found. Computing one from training images\n')
#             vocab_sizes = [50, 100, 200, 400, 800]
#             print("Finding optimal vocabulary size using the validation set")
#             best_vocab_size, accuracies = find_optimal_vocab_size(
#             train_image_paths, train_labels, val_image_paths, val_labels, vocab_sizes
#             )
        
#             # Plot accuracy vs. vocabulary size
#             plt.plot(vocab_sizes, accuracies, marker='o')
#             plt.title('Accuracy vs. Vocabulary Size')
#             plt.xlabel('Vocabulary Size')
#             plt.ylabel('Accuracy')
#             plt.grid()
#             plt.savefig('Accuracy_vs_Vocabsize.png')
#             print(f"Optimal vocabulary size: {best_vocab_size}")

#             # Build vocabulary with the optimal size
#             print(f"Building vocabulary with size: {best_vocab_size}")
#             vocab = build_vocabulary(train_image_paths, best_vocab_size)
#             with open('vocab.pkl', 'wb') as handle:
#                 pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

#             # Extract features for train and test sets
#             print("Extracting features for training and testing sets")
#             if os.path.isfile('train_image_feats.pkl') is False:
#                 train_image_feats = get_bags_of_sifts(train_image_paths, vocab)
#                 with open('train_image_feats.pkl', 'wb') as handle:
#                     pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
#             else:
#                 with open('train_image_feats.pkl', 'rb') as handle:
#                     train_image_feats = pickle.load(handle)
            
#             if os.path.isfile('test_image_feats.pkl') is False:
#                 test_image_feats = get_bags_of_sifts(test_image_paths, vocab)
#                 with open('test_image_feats.pkl', 'wb') as handle:
#                     pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
#             else:
#                 with open('test_image_feats.pkl', 'rb') as handle:
#                     test_image_feats = pickle.load(handle)

#     elif FEATURE == 'tiny_image':
#         print("Using tiny images as features")
#         train_image_feats = get_tiny_images(train_image_paths)
#         test_image_feats = get_tiny_images(test_image_paths)

#     elif FEATURE == 'dumy_feature':
#         train_image_feats = []
#         test_image_feats = []
#     else:
#         raise NameError('Unknown feature type')

#     # Step 2: Classify test images and evaluate accuracy
#     if CLASSIFIER == 'nearest_neighbor':
#         print("Classifying using Nearest Neighbor")
#         predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)
#     elif CLASSIFIER == 'support_vector_machine':
#         print("Classifying using Support Vector Machine")
#         predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
#     elif CLASSIFIER == 'dumy_classifier':
#         print("Using Dummy Classifier")
#         predicted_categories = test_labels[:]
#         shuffle(predicted_categories)
#     else:
#         raise NameError('Unknown classifier type')

#     # Compute and print overall accuracy
#     accuracy = compute_accuracy(predicted_categories, test_labels)
#     print(f"Test Accuracy: {accuracy:.2f}")

#     # Compute and print class-wise accuracy
#     for category in CATEGORIES:
#         category_accuracy = compute_accuracy(predicted_categories, test_labels, category)
#         print(f"{category}: {category_accuracy:.2f}")

#     # Step 3: Build and display confusion matrix
#     print("Building confusion matrix")
#     test_labels_ids = [CATE2ID[x] for x in test_labels]
#     predicted_categories_ids = [CATE2ID[x] for x in predicted_categories]

#     build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES)

#     # Step 4: Perform t-SNE visualization of SIFT descriptors
#     if FEATURE == 'bag_of_sift':
#         print("Performing t-SNE visualization")
#         _, all_train_descriptors = get_bags_of_sifts(train_image_paths, vocab)
#         visualize_tsne(all_train_descriptors, train_labels)

def visualize_tsne(descriptors, labels):
    flattened_descriptors = np.vstack(descriptors)
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(flattened_descriptors)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='tab10', s=5)
    plt.colorbar()
    plt.title('t-SNE Visualization of SIFT Descriptors')
    plt.savefig("tsne_visulaization.png")


# def main():
#     #This function returns arrays containing the file path for each train
#     #and test image, as well as arrays with the label of each train and
#     #test image. By default all four of these arrays will be 1500 where each
#     #entry is a string.
#     print("Getting paths and labels for all train and test data")
#     train_image_paths, train_labels = get_image_paths(TRAIN_PATH, CATEGORIES)
#     val_image_paths, val_labels = get_image_paths(VAL_PATH, CATEGORIES)
#     test_image_paths, test_labels = get_image_paths(TEST_PATH, CATEGORIES)

#     # TODO Step 1:
#     # Represent each image with the appropriate feature
#     # Each function to construct features should return an N x d matrix, where
#     # N is the number of paths passed to the function and d is the 
#     # dimensionality of each image representation. See the starter code for
#     # each function for more details.

#     if FEATURE == 'tiny_image':
#         # YOU CODE get_tiny_images.py 
#         train_image_feats = get_tiny_images(train_image_paths)
#         test_image_feats = get_tiny_images(test_image_paths)

#     elif FEATURE == 'bag_of_sift':
#         # YOU CODE build_vocabulary.py
#         if os.path.isfile('vocab.pkl') is False:
#             print('No existing visual word vocabulary found. Computing one from training images\n')
#             vocab_size = 400   ### Vocab_size is up to you. Larger values will work better (to a point) but be slower to comput.
#             vocab = build_vocabulary(train_image_paths, vocab_size)
#             with open('vocab.pkl', 'wb') as handle:
#                 pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

#         if os.path.isfile('train_image_feats_1.pkl') is False:
#             # YOU CODE get_bags_of_sifts.py
#             train_image_feats = get_bags_of_sifts(train_image_paths);
#             with open('train_image_feats_1.pkl', 'wb') as handle:
#                 pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         else:
#             with open('train_image_feats_1.pkl', 'rb') as handle:
#                 train_image_feats = pickle.load(handle)

#         if os.path.isfile('test_image_feats_1.pkl') is False:
#             test_image_feats  = get_bags_of_sifts(test_image_paths);
#             with open('test_image_feats_1.pkl', 'wb') as handle:
#                 pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         else:
#             with open('test_image_feats_1.pkl', 'rb') as handle:
#                 test_image_feats = pickle.load(handle)
#     elif FEATURE == 'dumy_feature':
#         train_image_feats = []
#         test_image_feats = []
#     else:
#         raise NameError('Unknown feature type')

#     # TODO Step 2: 
#     # Classify each test image by training and using the appropriate classifier
#     # Each function to classify test features will return an N x 1 array,
#     # where N is the number of test cases and each entry is a string indicating
#     # the predicted category for each test image. Each entry in
#     # 'predicted_categories' must be one of the 15 strings in 'categories',
#     # 'train_labels', and 'test_labels.

#     if CLASSIFIER == 'nearest_neighbor':
#         # YOU CODE nearest_neighbor_classify.py
#         predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)

#     elif CLASSIFIER == 'support_vector_machine':
#         # YOU CODE svm_classify.py
#         predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)

#     elif CLASSIFIER == 'dumy_classifier':
#         # The dummy classifier simply predicts a random category for
#         # every test case
#         predicted_categories = test_labels[:]
#         shuffle(predicted_categories)
#     else:
#         raise NameError('Unknown classifier type')

#     accuracy = float(len([x for x in zip(test_labels,predicted_categories) if x[0]== x[1]]))/float(len(test_labels))
#     print("Accuracy = ", accuracy)
    
#     for category in CATEGORIES:
#         accuracy_each = float(len([x for x in zip(test_labels,predicted_categories) if x[0]==x[1] and x[0]==category]))/float(test_labels.count(category))
#         print(str(category) + ': ' + str(accuracy_each))
    
#     test_labels_ids = [CATE2ID[x] for x in test_labels]
#     predicted_categories_ids = [CATE2ID[x] for x in predicted_categories]
#     train_labels_ids = [CATE2ID[x] for x in train_labels]
    
#     # Step 3: Build a confusion matrix and score the recognition system
#     # You do not need to code anything in this section. 
   
#     build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES)
#     visualize(CATEGORIES, test_image_paths, test_labels_ids, predicted_categories_ids, train_image_paths, train_labels_ids)

def compute_accuracy(predicted_labels, true_labels):
    """
    Computes the accuracy of predictions.
    
    Args:
        predicted_labels (list): Predicted categories for the data.
        true_labels (list): True categories for the data.

    Returns:
        float: Accuracy as a fraction (0 to 1).
    """
    correct_predictions = sum(p == t for p, t in zip(predicted_labels, true_labels))
    total_predictions = len(true_labels)
    return correct_predictions / total_predictions if total_predictions > 0 else 0.0

def find_optimal_vocab_size(train_paths, train_labels, val_paths, val_labels, vocab_sizes):
    best_vocab_size = None
    best_accuracy = 0
    accuracies = []

    for vocab_size in vocab_sizes:
        print(f"Testing with vocabulary size: {vocab_size}")
        vocab = build_vocabulary(train_paths, vocab_size)

        train_feats = get_bags_of_sifts(train_paths, vocab)
        val_feats = get_bags_of_sifts(val_paths, vocab)

        predicted_labels = svm_classify(train_feats, train_labels, val_feats)
        accuracy = compute_accuracy(predicted_labels, val_labels)
        accuracies.append(accuracy)

        print("Finding the best vocabulary size")
        print(f"Accuracy with vocab size {vocab_size}: {accuracy:.2f}")
        if accuracy > best_accuracy:
            best_vocab_size = vocab_size
            best_accuracy = accuracy

    return best_vocab_size, accuracies

def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories):
    # Compute confusion matrix
    cm = confusion_matrix(test_labels_ids, predicted_categories)
    np.set_printoptions(precision=2)
    '''
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, CATEGORIES)
    '''
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print('Normalized confusion matrix')
    #print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, abbr_categories, title='Normalized confusion matrix')

    plt.show()
     
def plot_confusion_matrix(cm, category, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    main()
