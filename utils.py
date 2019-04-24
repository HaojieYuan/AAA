import os
import torch
import numpy as np
from models.fresnet import r100_basic

def remove_prefix_string(string, prefix):
    assert string.startswith(prefix), "can not remove prefix."
    return string[len(prefix):]

def remove_prefix_from_state_dict(state_dict, prefix):
    for old_key in list(state_dict.keys()):
        if old_key.startswith(prefix):
            new_key = remove_prefix_string(old_key, prefix)
            state_dict[new_key] = state_dict.pop(old_key)

def load_state_local(path, model, ignore=[], optimizer=None):
    def map_func(storage, location):
        return storage.cuda()
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        if len(ignore) > 0:
            assert optimizer == None
            for k in list(checkpoint['state_dict'].keys()):
                flag = False
                for prefix in ignore:
                     if k.startswith(prefix):
                         flag = True
                         the_prefix = prefix
                         break
                if flag:
                    print('ignoring {} (prefix: {})'.format(k, the_prefix))
                    del checkpoint['state_dict'][k]
        remove_prefix_from_state_dict(checkpoint['state_dict'], 'module.base.')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        keys1 = set(checkpoint['state_dict'].keys())
        keys2 = set([k for k,_ in model.named_parameters()])
        not_loaded = keys2 - keys1
        for k in not_loaded:
            print('caution: {} not loaded'.format(k))
        if optimizer != None:
            assert len(ignore) == 0
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (step {})".format(path, checkpoint['step']))
            return checkpoint['step']
    else:
        assert False, "=> no checkpoint found at '{}'".format(path)

def load_face_model():
    """ load a face embedding model for attack. """
    ckpt_path = "/tf/FaceAdvTransfer/face_model/experiments/ArcFace/checkpoints/ckpt_task0_iter_38000.pth.tar"
    model = r100_basic(512)
    load_state_local(ckpt_path, model)
    return model
    
def image2tensor(image_obj, batch_dim=True):
    # if batch_dim, return 4 dim image tensor.
    np_img = np.array(image_obj)
    if batch_dim:
        np_img = np_img.reshape((-1,*np_img.shape))
        t_img = torch.from_numpy(np_img)
        return t_img.permute(0, 3, 1, 2).to(dtype=torch.float32)
    else:
        t_img = torch.from_numpy(np_img)
        return t_img.permute(2, 0, 1).to(dtype=torch.float32)


