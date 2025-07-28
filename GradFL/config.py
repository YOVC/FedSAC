import yaml

global cfg
if 'cfg' not in globals():
    with open('config.yml', 'r') as file:
        cfg = yaml.safe_load(file)

if __name__ == '__main__':
    print(cfg)