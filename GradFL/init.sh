#!/usr/bin/bash
if [[ -d "./data" ]]; then
  echo "The data dir have existed!"
else
  mkdir "./data"
fi

cd data

# Download emnist
if [[ -d "./emnist" ]]; then
  echo "./data/emnist have existed!"
else
  mkdir "./emnist"
  cd emnist
  emnist_url[0]="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
  emnist_url[1]="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
  emnist_url[2]="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
  emnist_url[3]="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

  file_count=$(ls -1A | wc -l)
  if [[ ${file_count} -eq 4 ]]; then
    echo "emnist have downloaded!"
  else
    for url in ${emnist_url[@]}
    do
      wget ${url}
    done
  fi

  for gz in $(ls)
  do
    gzip -d ${gz}
  done
  cd ..
fi

# Download cifar10
if [[ -d "./cifar10" ]]; then
  echo "cifar10 have existed!"
else
  if [[ -f "./cifar-10-python.tar.gz" ]]; then
    echo "cifar10.tar.gz have existed!"
  else
    wget "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
  fi
  tar -xvf cifar-10-python.tar.gz
  mv  cifar-10-batches-py cifar10
fi

# Download cifar100
if [[ -d "./cifar100" ]]; then
  echo "cifar100 have existed!"
else
  if [[ -f "./cifar-100-python.tar.gz" ]]; then
    echo cifar10.tar.gz have existed!
  else
    wget "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
  fi
  tar -xvf cifar-100-python.tar.gz
  mv cifar-100-python cifar100
fi

# Download tinyiamgenet
if [[ -d "./tinyimagenet" ]]; then
  echo tinyimagenet have exisited!
else
  mkdir "./tinyimagenet"
  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download --repo-type dataset --resume-download zh-plus/tiny-imagenet --local-dir ./tinyimagenet --local-dir-use-symlinks False
fi
