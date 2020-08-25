import datasets.cub200
import datasets.cars196
import datasets.stanford_online_products
import datasets.inshop


def select(dataset, opt, data_path):
    if 'cub200' in dataset:
        return cub200.Give(opt, data_path)

    if 'cars196' in dataset:
        return cars196.Give(opt, data_path)

    if 'online_products' in dataset:
        return stanford_online_products.Give(opt, data_path)

    if 'in-shop' in dataset:
        return inshop.Give(opt, data_path)

    raise NotImplementedError('A dataset for {} is currently not implemented.\n\
                               Currently available are : cub200, cars196, stanford_online_products & in-shop!'.format(dataset))
