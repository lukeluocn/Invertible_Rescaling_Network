'''create dataset and dataloader'''
import logging
# import torch
# import torch.utils.data
import os
import mindspore
import mindspore.dataset as de



# def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
#     phase = dataset_opt['phase']
#     if phase == 'train':
#         if opt['dist']:
#             world_size = torch.distributed.get_world_size()
#             num_workers = dataset_opt['n_workers']
#             assert dataset_opt['batch_size'] % world_size == 0
#             batch_size = dataset_opt['batch_size'] // world_size
#             shuffle = False
#         else:
#             num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
#             batch_size = dataset_opt['batch_size']
#             shuffle = True
#         return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
#                                            num_workers=num_workers, sampler=sampler, drop_last=True,
#                                            pin_memory=False)
#     else:
#         return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
#                                            pin_memory=True)


def create_dataset(phase, dataset_opt,gpu_ids):
    mode = dataset_opt['mode']
    if mode == 'LQ':
        from data.LQ_dataset import LQDataset as D
    elif mode == 'LQGT':
        from data.LQGT_dataset import LQGTDataset as D
    # elif mode == 'LQGTseg_bg':
    #     from data.LQGT_seg_bg_dataset import LQGTSeg_BG_Dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))

    if phase == "train":
        dataset = D(dataset_opt)
        sampler = None
        num_workers = dataset_opt['n_workers'] * len(gpu_ids)
        shuffle = True
        de_dataset =  de.GeneratorDataset(dataset,["low_quality","ground_truth"],sampler=sampler,
                                    num_parallel_workers=num_workers,shuffle=shuffle)  

        batch_size = dataset_opt['batch_size']
        de_dataset=de_dataset.batch(batch_size)

        columns_to_project = ["low_quality","ground_truth"]
        de_dataset = de_dataset.project(columns=columns_to_project)
        
        logger = logging.getLogger('base')
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                            dataset_opt['name']))
        return de_dataset
    else:
        dataset = D(dataset_opt)
        sampler = None
        num_workers = 1
        shuffle = False
        de_dataset =  de.GeneratorDataset(dataset,["low_quality","ground_truth"],sampler=sampler,
                                    num_parallel_workers=num_workers,shuffle=shuffle) 

        batch_size = 1
        de_dataset=de_dataset.batch(batch_size)

        columns_to_project = ["low_quality","ground_truth"]
        de_dataset = de_dataset.project(columns=columns_to_project)

        logger = logging.getLogger('val')
        logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(dataset)))
        return de_dataset


