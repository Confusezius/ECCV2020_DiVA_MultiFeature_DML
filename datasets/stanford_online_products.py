from datasets.basic_dataset_scaffold import BaseDataset
import os
import pandas as pd


def Give(opt, datapath):
    image_sourcepath  = opt.source_path+'/images'
    training_files = pd.read_table(opt.source_path+'/Info_Files/Ebay_train.txt', header=0, delimiter=' ')
    test_files     = pd.read_table(opt.source_path+'/Info_Files/Ebay_test.txt', header=0, delimiter=' ')


    conversion, super_conversion = {},{}
    for class_id, path in zip(training_files['class_id'],training_files['path']):
        conversion[class_id] = path.split('/')[0]
    for super_class_id, path in zip(training_files['super_class_id'],training_files['path']):
        conversion[super_class_id] = path.split('/')[0]
    for class_id, path in zip(test_files['class_id'],test_files['path']):
        conversion[class_id] = path.split('/')[0]

    train_image_dict, test_image_dict, super_train_image_dict  = {},{},{}
    for key, img_path in zip(training_files['class_id'],training_files['path']):
        key = key-1
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(image_sourcepath+'/'+img_path)

    for key, img_path in zip(test_files['class_id'],test_files['path']):
        key = key-1
        if not key in test_image_dict.keys():
            test_image_dict[key] = []
        test_image_dict[key].append(image_sourcepath+'/'+img_path)

    for key, img_path in zip(training_files['super_class_id'],training_files['path']):
        key = key-1
        if not key in super_train_image_dict.keys():
            super_train_image_dict[key] = []
        super_train_image_dict[key].append(image_sourcepath+'/'+img_path)


    train_keys  = list(train_image_dict.keys())

    if opt.train_val_split!=1:
        if opt.train_val_split_by_class:
            train_val_split = int(len(train_keys)*opt.train_val_split)
            train, val  = train_keys[:train_val_split], train_keys[train_val_split:]
            train_image_dict, val_image_dict = {key:train_image_dict[key] for key in train}, {key:train_image_dict[key] for key in val}
        else:
            train_image_dict_temp, val_image_dict_temp = {},{}
            for key in train_keys:
                train_ixs = np.random.choice(len(train_image_dict[key]), int(len(train_image_dict[key])*opt.train_val_split), replace=False)
                val_ixs   = np.array([x for x in range(len(train_image_dict[key])) if x not in train_ixs])
                train_image_dict_temp[key] = np.array(image_dict[key])[train_ixs]
                val_image_dict_temp[key]   = np.array(image_dict[key])[val_ixs]
        val_dataset = BaseDataset(val_image_dict,   opt, is_validation=True)
        val_dataset.conversion = conversion
    else:
        val_dataset = None

    super_train_dataset = BaseDataset(super_train_image_dict, opt, is_validation=True)
    train_dataset       = BaseDataset(train_image_dict, opt)
    test_dataset        = BaseDataset(test_image_dict,  opt, is_validation=True)
    eval_dataset        = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset  = BaseDataset(train_image_dict, opt)

    super_train_dataset.conversion = super_conversion
    train_dataset.conversion       = conversion
    test_dataset.conversion        = conversion
    eval_dataset.conversion        = conversion

    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'evaluation_train':eval_train_dataset, 'super_evaluation':super_train_dataset}
