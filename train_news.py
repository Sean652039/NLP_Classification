import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from data_cnews import CnewsDataset
from vocab import Vocab
from model_news import MyLSTM, read_pretrained_wordvec
from tqdm import tqdm
import numpy as np
import random
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Text Classification Training Script')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--vocab_size', type=int, default=6000, help='Vocabulary size')
    parser.add_argument('--padding_len', type=int, default=500, help='Padding length for sequences')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='mps',
                        help='Device for training and evaluation (cpu, cuda, or mps)')

    return parser.parse_args()


def train(epoch, net):
    """
 net.train()和net.eval()到底在什么时候使用？
 如果一个模型有Dropout与BatchNormalization，
 那么它在训练时要以一定概率进行Dropout或者更新BatchNormalization参数，
 而在测试时不在需要Dropout或更新BatchNormalization参数。此时，要用net.train()和net.eval()进行区分。
    """
    """
 When exactly are net.train() and net.eval() used?
 If a model has Dropout & BatchNormalization.
 Then it has to Dropout or update the BatchNormalization parameter with a certain probability during training.
 It does not need to Dropout or update the BatchNormalization parameter during testing. In this case, 
 it is important to distinguish between net.train() and net.eval().
    """
    def evaluate(curr_ep):
        net.eval()
        correct = 0
        all = 0
        with torch.no_grad():   # 逃避autograd的追踪,因为评估和测试数据不需要计算梯度，也不会进行反向传播
                                # Evade autograd tracking, as evaluation and test data do not need to compute gradients
                                # and do not backpropagate
            for (x, y) in tqdm(val_dataset):
                x, y = x.to(device), y.to(device)
                logits = net(x)
                logits = torch.argmax(logits, dim=-1)
                correct += torch.sum(logits.eq(y)).float().item()
                all += y.size()[0]
                tensorboard_writer.add_scalar('Validation/Acc', correct / all, curr_ep)
        print(f'evaluate done! acc {correct / all:.5f}')

    is_model_stored = False
    for ep in range(epoch):
        print(f'epoch {ep} start')
        net.train()
        for (x, y) in tqdm(train_dataset):
            x, y = x.to(device), y.to(device)

            # 用第一个训练数据作为模型输入来储存模型
            # Use the first training data as model input to store the model
            if not is_model_stored:
                tensorboard_writer.add_graph(net, x)
                is_model_stored = True

            # 前向传播求出预测的值,执行net中的forward函数
            # Forward propagation to find the predicted value, execute the forward function in net.
            logits = net(x)

            # loss
            loss = criterion(logits, y)

            # 在训练过程中记录训练损失
            # Record training losses during training
            tensorboard_writer.add_scalar('Train/Loss', loss.item(), ep)

            """"
            将梯度初始化为零。因为训练的过程通常使用mini-batch方法，所以如果不将梯度清零的话，
            梯度会与上一个batch的数据相关，因此该函数要写在反向传播和梯度下降之前。
            Initialize the gradient to zero. Because the training process usually uses the mini-batch method, 
            the gradient will be correlated with the data from the previous batch if it is not zeroed out, 
            so this function is written before backpropagation and gradient descent.
            """
            optimizer.zero_grad()

            """
            反向传播求梯度。损失函数loss是由模型的所有权重w经过一系列运算得到的，
            若某个w的requires_grads为True，则w的所有上层参数（后面层的权重w）的.grad_fn属性中就保存了对应的运算，
            然后在使用loss.backward()后，会一层层的反向传播计算每个w的梯度值，并保存到该w的.grad属性中。
            Backpropagation for gradients. The loss function loss is obtained 
            from all the weights w of the model through a series of operations. 
            If the requires_grads of a certain w is True, 
            the corresponding operations are saved in the .grad_fn attribute of all the upper parameters 
            of the w (the weights w of the later layers), and then after using loss.backward(), 
            it will backpropagate one layer at a time to calculate the gradient value of each w 's gradient value 
            and saves it in that w's .grad attribute.
            """
            loss.backward()

            """
            更新所有参数。step()函数的作用是执行一次优化步骤，
            通过梯度下降法来更新参数的值。optimizer只负责通过梯度下降进行优化，而不负责产生梯度，梯度是tensor.backward()方法产生的。
            Update all the parameters. step() function is used to perform one optimization step to update the values of 
            the parameters by gradient descent method. 
            optimizer is only responsible for optimization by gradient descent and not responsible for generating the gradient, 
            which is generated by tensor.backward() method.
            """
            optimizer.step()

        evaluate(ep)


def test(net):
        net.eval()
        correct = 0
        all = 0

        with torch.no_grad():
            for (x, y) in tqdm(test_dataset):
                x, y = x.to(device), y.to(device)
                logits = net(x)
                logits = torch.argmax(logits, dim=-1)
                correct += torch.sum(logits.eq(y)).float().item()
                all += y.size()[0]
        print(f'test done! acc {correct / all:.5f}')


if __name__ == '__main__':
    args = parse_args()

    # 算力设备，支持Apple Silicon
    # Device selection based on args
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda:0' if torch.cuda.device_count() > 1 else 'cuda')
    else:
        device = torch.device('cpu')

    # 创建“dataset”文件夹
    # Create a 'dataset' folder if it doesn't exist
    dataset_folder = 'dataset'
    os.makedirs(dataset_folder, exist_ok=True)

    # 定义保存数据的路径
    # Define file paths for saved datasets and vocabulary
    train_dataset_path = os.path.join(dataset_folder, 'cnews.train.pth')
    val_dataset_path = os.path.join(dataset_folder, 'cnews.val.pth')
    test_dataset_path = os.path.join(dataset_folder, 'cnews.test.pth')

    """
    这是将转换为token的序列保存，在每次训练前直接读取，减少程序运行时间。
    This is done by saving the sequence converted to a token and reading it directly before each training session, 
    reducing program runtime.
    """
    # 检查处理过的数据集和词汇文件是否存在
    # Check if processed datasets and vocabulary files exist
    if os.path.exists(train_dataset_path) and os.path.exists(val_dataset_path) \
            and os.path.exists(test_dataset_path):
        # 加载处理过的数据集和词汇表
        # Load processed datasets and vocabulary
        train_dataset = torch.load(train_dataset_path)
        val_dataset = torch.load(val_dataset_path)
        test_dataset = torch.load(test_dataset_path)
    else:
        train_dataset = CnewsDataset(r'cnews/cnews.train.txt')
        val_dataset = CnewsDataset(r'cnews/cnews.val.txt')
        test_dataset = CnewsDataset(r'cnews/cnews.test.txt')

        # 保存处理过的数据集和词汇表
        # Save processed datasets and vocabulary
        torch.save(train_dataset, train_dataset_path)
        torch.save(val_dataset, val_dataset_path)
        torch.save(test_dataset, test_dataset_path)

    vocab = Vocab(train_dataset.inputs, args.vocab_size)
    train_dataset.token2seq(vocab, args.padding_len)
    val_dataset.token2seq(vocab, args.padding_len)
    test_dataset.token2seq(vocab, args.padding_len)
    train_dataset = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataset = DataLoader(test_dataset, batch_size=args.batch_size)

    # 设置随机数种子
    # random seed
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 创建 SummaryWriter，指定记录的目录，记录训练过程的数据
    # Create a SummaryWriter that specifies the directory where the records will be logged and records the data from the training process
    tensorboard_writer = SummaryWriter('logs')

    net = MyLSTM(read_pretrained_wordvec(r'words/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt', vocab, 100),
                 len(vocab), 100, 2, 128, 10)

    net = net.to(device)  # 网络布置到设备上计算  # Network placement to device calculations
    optimizer = optim.Adam(net.parameters(), lr = args.lr)  # 可以换成SGD  # Can be replaced by SGD
    criterion = nn.CrossEntropyLoss().to(device)

    train(args.epochs, net)
    test(net)

    tensorboard_writer.close()

