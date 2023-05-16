import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--control', type=int, default=10, help='control factor, threads = cpu_num/control')
    parser.add_argument('--log_interval', type=int, default=20, help='interval for evaluation')
    parser.add_argument('--DS', dest='DS', help='Dataset')
    parser.add_argument('--epochs', dest='epochs', default = 20, type = int, help='epochs')
    parser.add_argument('--local', dest='local', action='store_const', 
            const=True, default=False)
    parser.add_argument('--glob', dest='glob', action='store_const', 
            const=True, default=False)
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=False)

    parser.add_argument('--lr', dest='lr', type=float,default = 0.01,
            help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
            help='')
    parser.add_argument('--device', dest='device', type=str, default='cpu',
                        help='')
    parser.add_argument('--a', type=float, default=0,
                        help='Gradient weight')
    return parser.parse_args()

