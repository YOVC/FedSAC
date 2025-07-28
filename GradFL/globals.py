import argparse
import re

pattern = r'\d+\.\d+|\d+'


def get_base_params():
    parser = argparse.ArgumentParser(description='cfg')
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--logname', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--frc', type=float)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--shardperuser', type=int)
    parser.add_argument('--device', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--probs', type=str)
    parser.add_argument('--inferen_batch', type=int)
    parser.add_argument('--select_model', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--group_name', type=str)
    parser.add_argument('--client_send_label', action='store_true', default=False)
    args = vars(parser.parse_args())
    if args['probs']:
        probs = re.findall(pattern, args['probs'])
        args['probs'] = [float(probs[0]), float(probs[1]), float(probs[2]), float(probs[3]), float(probs[4])]
    return args


if __name__ == '__main__':
    args = get_base_params()
    print(args)
