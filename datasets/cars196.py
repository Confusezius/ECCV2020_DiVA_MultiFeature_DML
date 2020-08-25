from datasets.basic_dataset_scaffold import BaseDataset
import os


def Give(opt, datapath):
    image_sourcepath  = datapath+'/images'
    image_classes = sorted([x for x in os.listdir(image_sourcepath)])
    conversion    = {i:x for i,x in enumerate(image_classes)}
    image_list    = {i:sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key)]) for i,key in enumerate(image_classes)}
    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    image_dict    = {}
    for key, img_path in image_list:
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)


    keys = sorted(list(image_dict.keys()))
    #Following "Deep Metric Learning via Lifted Structured Feature Embedding", we use the first half of classes for training.
    train,test      = keys[:len(keys)//2], keys[len(keys)//2:]

    if opt.train_val_split!=1:
        if opt.train_val_split_by_class:
            train_val_split = int(len(train)*opt.train_val_split)
            train, val      = train[:train_val_split], train[train_val_split:]
            train_image_dict, val_image_dict, test_image_dict = {key:image_dict[key] for key in train}, {key:image_dict[key] for key in val}, {key:image_dict[key] for key in test}
        else:
            train_image_dict, val_image_dict = {},{}
            for key in train:
                train_ixs = np.random.choice(len(image_dict[key]), int(len(image_dict[key])*opt.train_val_split), replace=False)
                val_ixs   = np.array([x for x in range(len(image_dict[key])) if x not in train_ixs])
                train_image_dict[key] = np.array(image_dict[key])[train_ixs]
                val_image_dict[key]   = np.array(image_dict[key])[val_ixs]
        val_dataset   = BaseDataset(val_image_dict,   opt, is_validation=True)
        val_dataset.conversion   = conversion
    else:
        train_image_dict = {key:image_dict[key] for key in train}
        val_dataset = None

    test_image_dict = {key:image_dict[key] for key in test}

    train_dataset = BaseDataset(train_image_dict, opt)
    test_dataset  = BaseDataset(test_image_dict,  opt, is_validation=True)
    eval_dataset  = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset  = BaseDataset(train_image_dict, opt, is_validation=False)
    train_dataset.conversion = conversion
    test_dataset.conversion  = conversion
    eval_dataset.conversion  = conversion

    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'evaluation_train':eval_train_dataset}
