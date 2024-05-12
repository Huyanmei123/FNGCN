import argparse
import torch

"""
hamster
proteins
ca-GrQc
ca-HepTh
SC
ca-CondMat

"""
args = argparse.ArgumentParser()
args.add_argument('--dataset', default='ca-CondMat')
args.add_argument('--model', default='GCN')
args.add_argument('--learning_rate', default=0.01)  # 0.01 for GCN 0.05 for GCNII
args.add_argument('--epochs', default=9999)
args.add_argument('--hidden_units', default=16)  # 16 for GCN 32 for GCNII
args.add_argument('--dropout', default=0.4)
args.add_argument('--weight_decay', default=0.01)
args.add_argument('--weight_decay1', default=0.01)
args.add_argument('--weight_decay2', default=0.0005)
args.add_argument('--early_stopping', default=30)  # 30 for GCN 100 for GCNII
args.add_argument('--residual', default=True)
args.add_argument('--seed', default=824)
args.add_argument('--type', default='Binary')
args.add_argument('--DEVICE', default="cuda" if torch.cuda.is_available() else "cpu")
args.add_argument('--theta', default=0.5)
args.add_argument('--layers', default=64)
args.add_argument('--alpha', default=0.1)
args.add_argument('--feature_type', default='x-2.2')
args = args.parse_args()

"""
feature_type 2.2 represents node features constructed by feature network
"""
