import numpy as np
import pdb
import shutil
import os

def visualize(CATEGORIES, test_image_paths, test_labels_ids, predicted_categories_ids, train_labels_paths, train_labels_ids):
    main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    thumbnails_path = os.path.join(main_path, 'results', 'thumbnails')
    if os.path.isdir(thumbnails_path):
        shutil.rmtree(thumbnails_path)
        os.makedirs(thumbnails_path)

    results_path = os.path.join(main_path, 'results')
    
    panel = open(os.path.join(results_path, 'visualizatoin.md'), 'w')
    panel.write('## Visualization\n')
    panel.write('| Category name | Sample training images | Sample true positives | False positives with true label | False negatives with wrong predicted label |\n')
    panel.write('| :-----------: | :--------------------: | :-------------------: | :-----------------------------: | :----------------------------------------: |\n')

    FN_name = [None] * len(CATEGORIES)
    TP_name = [None] * len(CATEGORIES)
    FP_name = [None] * len(CATEGORIES)
    Train_name = [None] * len(CATEGORIES)

    for k, name in enumerate(CATEGORIES):
        train_id = np.where(np.array(train_labels_ids) == k)[0].tolist()
        Train_name[k] = train_labels_paths[train_id[0]]
        instance_id = np.where(np.array(test_labels_ids) == k)
        instance_id = instance_id[0].tolist()
        instance_name = [test_image_paths[x] for x in instance_id]
        result = [predicted_categories_ids[x] for x in instance_id]
        for sub_id, sub_pred in enumerate(result):
            if sub_pred != k:
                FN_name[k] = instance_name[sub_id]
            elif sub_pred == k:
                TP_name[k] = instance_name[sub_id]

        pred_instance_id = np.where(np.array(predicted_categories_ids) == k)
        pred_instance_id = pred_instance_id[0].tolist()
        pred_instance_name = [test_image_paths[x] for x in pred_instance_id]
        pred_result = [test_labels_ids[x] for x in pred_instance_id]
        for sub_id, sub_pred in enumerate(pred_result):
            if sub_pred != k:
                FP_name[k] = pred_instance_name[sub_id]
        shutil.copy(Train_name[k], os.path.join(thumbnails_path, name + '_train_' + os.path.basename(Train_name[k])))
        shutil.copy(TP_name[k], os.path.join(thumbnails_path, name + '_TP_' + os.path.basename(TP_name[k])))
        shutil.copy(FP_name[k], os.path.join(thumbnails_path, name + '_FP_' + os.path.basename(FP_name[k])))
        shutil.copy(FN_name[k], os.path.join(thumbnails_path, name + '_FN_' + os.path.basename(FN_name[k])))
        train_path = os.path.relpath(os.path.join(thumbnails_path, name + '_train_' + os.path.basename(Train_name[k])), results_path)
        tp_path = os.path.relpath(os.path.join(thumbnails_path, name + '_TP_' + os.path.basename(TP_name[k])), results_path)
        fp_path = os.path.relpath(os.path.join(thumbnails_path, name + '_FP_' + os.path.basename(FP_name[k])), results_path)
        fn_path = os.path.relpath(os.path.join(thumbnails_path, name + '_FN_' + os.path.basename(FN_name[k])), results_path)
        panel.write('| ' + name + ' | ' + '![]('+train_path+')' + ' | ' + '![]('+tp_path+')' + ' | ' + '![]('+fp_path+')' + ' | ' + '![]('+fn_path+')' + ' |' + '\n')
        
    panel.write('\n')
    panel.close()
