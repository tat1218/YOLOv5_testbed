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
    parser.add_argument('--start', type=int, default=0)
    #parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--server_addr', type=str, default='localhost', help='server IP address')

    args = parser.parse_args()

    print("Waiting...")
    os.environ['MASTER_ADDR'] = args.server_addr
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('Gloo', rank=args.rank, world_size=2)

    m = yolo.Model(cfg="./models/yolov5s.yaml")
    
    y = [torch.empty(sz[y_idx]) if y_idx in y_required else None for y_idx in range(23)]

    for y_idx in y_required:
        if y_idx >= args.start:
            break
        print("wait... ", y_idx, sz[y_idx])
        dist.recv(y[y_idx], src=1)
        print("recv : ",y_idx)   

    x = torch.empty(sz[args.start-1], dtype=torch.float32)

    print("wait... ", args.start-1, sz[args.start-1])
    dist.recv(x,src=1)
    print("recv x : ", x.shape)
    
    ret = m(x,y=y,si=args.start,ei=-1)
    print(ret)
