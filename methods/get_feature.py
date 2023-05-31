import numpy as np
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm

def get_feature(net, testloader, outloader, **options):
    net.eval()
    test_feat, test_labels = [], []
    out_feat , out_labels = [], []
    torch.cuda.empty_cache()
    with torch.no_grad():
        for data, labels, _ in tqdm(testloader):
            if options['use_gpu']:
                data = data.cuda()
            with torch.set_grad_enabled(False):
                x = net(data)   
                feat_batch = x.cpu().numpy()
                test_feat.append(feat_batch)
                test_labels.append(labels)

        for data, labels, _ in tqdm(outloader):
            if options['use_gpu']:
                data = data.cuda()
            with torch.set_grad_enabled(False):
                x = net(data)   
                feat_batch = x.cpu().numpy()
                out_feat.append(feat_batch)
                out_labels.append(labels)

        test_feat = np.concatenate(test_feat, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        out_feat = np.concatenate(out_feat, axis=0)
        out_labels = np.concatenate(out_labels, axis=0)

        post_name = '_{}_{}.pkl'.format(options['loss'], options['dataset'])
        with open('./features/in_feature'+post_name, 'wb') as f:
            pickle.dump(test_feat, f)
        with open('./features/out_feature'+post_name, 'wb') as f:
            pickle.dump(out_feat, f)
        with open('./features/in_labels'+post_name, 'wb') as f:
            pickle.dump(test_labels, f)
        with open('./features/out_labels'+post_name, 'wb') as f:
            pickle.dump(out_labels, f)

        print("Finshed to extract feature!")
