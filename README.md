Code for the paper:
> Yinjun Wu, Edgar Dobriban, Susan Davidson. "DeltaGrad: Rapid retraining of machine learning models" (ICML 2020)



Usage (go to the folder '/src'):

run MNIST with our methods (assume that we use regularized logistic regression for training):


1. preprocessing the dataset (--model: model name used for training):

python3 generate_dataset_train_test.py --model Logistic_regression --dataset MNIST


2. randomly generate deleted/added samples from the training dataset:

python3 generate_rand_delta_ids.py --dataset MNIST --ratio 0.001 --restart



3. start training phase on the full tranining dataset 
(--add: flags to specify for model updates after small additions,otherwise we do small deletions;  --train: flag to specify to train on the full training dataset, otherwise, we update the pretrained model; --bz: mini-batch size; --epochs: number of epochs for running SGD; --wd: l2 regularization coefficient or weight decay rate; --lr, --lrlen: use certain learning rate for how many epochs, the following command means that the learning rates at the first 10 epochs and the last 10 epochs are 0.1 and 0.05 respectively):
python3 main.py --bz 16384 --epochs 20 --model Logistic_regression --dataset MNIST --wd 0.0001  --lr 0.1 0.05  --lrlen 10 10  --train


4. update the mode after the training phase with the baseline method (retrain from the scratch, --add: flags to specify for model updates after small additions,otherwise we do small deletions):
python3 main.py --bz 16384 --epochs 20 --model Logistic_regression --dataset MNIST --wd 0.0001  --lr 0.1 0.05  --lrlen 10 10  --method baseline


output (similar to this, the first line is the time in seconds to update the model with the baseline method  while the second line is the difference between the model before deletions or additions and the updated model by baseline method):

time_baseline:: 4.629846096038818
model difference (l2 norm): tensor(0.0027, dtype=torch.float64, grad_fn=<SqrtBackward>)


5. update the model after the training phase with deltagrad (--add: flags to specify for model updates after small additions,otherwise we do small deletions; --init: the hyperparameter j_0 used in Deltagrad, -m: the hyperparameter m used in Deltagrad; --period: the hyperparameter T_0 used in Deltagrad; --cached_size: control how many history parameters or gradients from the training phase before deletions or additions pre-cached in the GPU for saving the IO overhead):
python3 main.py --bz 16384 --epochs 20 --model Logistic_regression --dataset MNIST --wd 0.0001  --lr 0.1 0.05  --lrlen 10 10  --method deltagrad --period 5 --init 20 -m 2 --cached_size 20


output (similar to this, the first line is the time in seconds to update the model while the second line is the difference between the updated model by deltagrad and the updated model by baseline method):
time_deltagrad:: 1.637620210647583
model difference (l2 norm): tensor(7.1898e-05, dtype=torch.float64, grad_fn=<SqrtBackward>)


