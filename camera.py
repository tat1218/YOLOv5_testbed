import torch
import numpy as np
import models.yolo as yolo
import torch.distributed as dist
import os
import argparse

sz = [
    (1, 32, 128, 128),      #0
    (1, 64, 64, 64),        #1
    (1, 64, 64, 64),        #2
    (1, 128, 32, 32),
    (1, 128, 32, 32),
    (1, 256, 16, 16),
    (1, 256, 16, 16),
    (1, 512, 8, 8),
    (1, 512, 8, 8),
    (1, 512, 8, 8),
    (1, 256, 8, 8),
    (1, 256, 16, 16),
    (1, 512, 16, 16),
    (1, 256, 16, 16),
    (1, 128, 16, 16),
    (1, 128, 32, 32),
    (1, 256, 32, 32),
    (1, 128, 32, 32),
    (1, 128, 16, 16),
    (1, 256, 16, 16),
    (1, 256, 16, 16),
    (1, 256, 8, 8),
    (1, 512, 8, 8)
]

y_required = [4,6,10,14,17,20,23]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0, help='rank')
    #parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    #dist.init_process_group('Gloo', rank=args.rank, world_size=2)

    m = yolo.Model(cfg="./models/yolov5s.yaml")
    img = torch.zeros((1,3,256,256))
    x = m(img, 0, args.end)
    #print(x.shape)
    for y_idx, y in enumerate(m.y):
        print("--------")
        if y != None:
            print("wait... ", y_idx, sz[y_idx])
            #dist.send(y,dst=0)
            print("send : ",y_idx, y.shape)
        else:
            print("None : ",y_idx)
    print("wait... ", args.end, sz[args.end])
    #dist.send(x,dst=0)
    print("send x : ", x.shape)
