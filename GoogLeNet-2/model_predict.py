import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import GoogLeNet, Inception
from torchvision.datasets import ImageFolder
from PIL import Image

if __name__ == "__main__":
    model = GoogLeNet(Inception)
    model.load_state_dict(torch.load('GoogLeNet-2/best_model.pth'))

    model = model.to("cuda" if torch.cuda.is_available() else 'cpu')
    classes = ['苹果', '香蕉', '葡萄', '橙子', '梨']
    image = Image.open('GoogLeNet-2/or.jfif')

    normalize = transforms.Normalize([0.229, 0.196,  0.143], [0.099, 0.080, 0.066])

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