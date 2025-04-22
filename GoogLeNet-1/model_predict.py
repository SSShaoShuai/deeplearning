import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import GoogLeNet, Inception
from torchvision.datasets import ImageFolder
from PIL import Image

if __name__ == "__main__":
    model = GoogLeNet(Inception)
    model.load_state_dict(torch.load('GoogLeNet-1/best_model.pth'))

    model = model.to("cuda" if torch.cuda.is_available() else 'cpu')
    classes = ['猫','狗']
    image = Image.open('GoogLeNet-1/cat1.jpg')

    normalize = transforms.Normalize([0.162, 0.151, 0.138],[0.058, 0.052, 0.048])

    # 定义数据处理方法变量
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    image = test_transform(image)

    # 添加批次维度
    image = image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        image = image.to("cuda" if torch.cuda.is_available() else 'cpu')
        output = model(image)
        pre_lab = torch.argmax(output, dim=1)
        result = pre_lab.item()
    print("预测值：", classes[result])