import argparse
import importlib

def read_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='name of dataset;', type=str, default='cifar10')
    parser.add_argument('--dist', help='type of distribution;', type=int, default=18)
    parser.add_argument('--skew', help='the degree of niid;', type=float, default=0)
    parser.add_argument('--num_clients', help='the number of clients;', type=int, default=10)

    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option

if __name__ == '__main__':
    option = read_option()
    TaskGen = getattr(importlib.import_module('.'.join(['benchmark', option['dataset'], 'core'])), 'TaskGen')
    generator = TaskGen(dist_id = option['dist'], skewness = option['skew'], num_clients=option['num_clients'])
    generator.run()

