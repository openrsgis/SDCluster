import os
from PIL import Image
from torch.utils.data import Dataset
from data.transforms import LeopartTransforms

class DataSet_iSAID_Pretrain(Dataset):
    def __init__(self, args):
        self.transform = LeopartTransforms(size_crops=args['transform']['size_crops'],
                                           nmb_crops=args['transform']['nmb_samples'],
                                           min_scale_crops=args['transform']["min_scale_crops"],
                                           max_scale_crops=args['transform']["max_scale_crops"],
                                           min_intersection=args['transform']["min_intersection_crops"],
                                           jitter_strength=args['transform']["jitter_strength"],
                                           blur_strength=args['transform']["blur_strength"]
                                           )
        self.data_list = data_list # The list of image paths 
        print('Read ' + str(len(self.data_list)) + ' images')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        img_path = self.data_list[item]
        img = Image.open(img_path)
        multi_crops, gc_bboxes, otc_bboxes, flags = self.transform(img)
        return multi_crops, gc_bboxes, otc_bboxes, flags
