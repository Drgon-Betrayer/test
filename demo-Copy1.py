import os
import argparse
import torch
import time

from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image
from utils.visualize import get_color_pallete

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fast_scnn',
                    help='model name (default: fast_scnn)')
parser.add_argument('--dataset', type=str, default='citys',
                    help='dataset name (default: citys)')
parser.add_argument('--weights-folder', default='./weights',
                    help='Directory for saving checkpoint models')
parser.add_argument('--inputpic', type=str,
                    default='./png',
                    help='path to the input picture')
parser.add_argument('--outdir', default='./test_result', type=str,
                    help='path to save the predict result')

parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.set_defaults(cpu=False)

args = parser.parse_args()


def getFileList(dir, Filelist, ext=None):
    """
        获取文件夹及其子文件夹中文件列表
        输入 dir：文件夹根目录
        输入 ext: 扩展名
        返回： 文件路径列表
        """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)
    return Filelist


def demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(device)
    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    # model load
    model = get_fast_scnn(args.dataset, pretrained=True, root=args.weights_folder, map_cpu=args.cpu).to(device)
    print('Finished loading model!')
    model.eval()
    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    imglist = getFileList(args.inputpic, [], 'png')
    for img in imglist:
        t0=time.time()
        image = Image.open(img).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        print(f'Done. ({time.time() - t0:.3f}s)')
        with torch.no_grad():
            outputs = model(image)
        print(f'Done. ({time.time() - t0:.3f}s)')
        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
        mask = get_color_pallete(pred, args.dataset)
        outname = os.path.splitext(os.path.split(img)[-1])[0] + '.png'
        mask.save(os.path.join(args.outdir, outname))
        print(f'Done. ({time.time() - t0:.3f}s)')
        print('*'*20)


if __name__ == '__main__':
    demo()
