import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet

def test_data_process():
    test_data = FashionMNIST(
        root = './data',
        train = False,
        transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
        download=True
    )

    test_dataloader = Data.DataLoader(
        dataset=test_data,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    return test_dataloader


def test_model_process(model, test_dataloader):
    # 设置测试所用到的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 将模型放入到训练设备中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        # 设置模型为评估模式
        model.eval()
        for test_data_x, test_data_y in test_dataloader:
            # 将特征放入到测试设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y = test_data_y.to(device)
            # 前向传播过程，输入为测试数据集，输出为对每个样本的预测值
            output = model(test_data_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则准确度加一
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本进行累加
            test_num += test_data_x.size(0)

    # 计算准确率
    test_acc = test_corrects.double() / test_num
    print("Test acc: {:.4f}".format(test_acc))


if __name__ == "__main__":
    model = LeNet()
    model.load_state_dict(torch.load("./model_save/best_model.pth"))
    # 加载测试数据
    test_dataloader = test_data_process()
    # 加载模型测试的函数
    test_model_process(model, test_dataloader)
