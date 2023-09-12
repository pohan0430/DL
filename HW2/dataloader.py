from torchvision.datasets import LSUN
from torchvision.transforms import Resize, Normalize
from torchvision.io import read_image
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from sys import argv
from tqdm import trange, tqdm
import os, os.path

class Lsun(Dataset):
    def __init__(self, base_path, maintain=True) -> None:
        super().__init__()

        self.data = []
        self.maintain = maintain
        self.normalizer = Normalize(0.5, 0.5)
        
        for name in tqdm(os.listdir(base_path), desc='[Loading]'):
            if self.maintain:
                self.data.append(self.normalizer(read_image(base_path + name)/255))
            else:
                self.data.append(base_path + name)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index] if self.maintain else self.normalizer(read_image(self.data[index])/255)

def get_loader(base_path, batch_size, num_workers, maintain=True):
    return DataLoader(
        Lsun(base_path, maintain),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
    )

if __name__ == '__main__':
    file_dir = argv.pop(0).split('dataloader.py')[0]

    if argv[0] == 'resize':
        size = (int(argv[1]), int(argv[2]))
        resizer = Resize(size)

        # lsun_set = LSUN(
        #     root=file_dir+'LSUN',
        #     classes=['church_outdoor_train'],
        #     transform=resizer
        # )
        image_path  = "/home/DL_HW2/ALL_images/images"
        images = [os.path.join(image_path, each) for each in os.listdir(image_path)]
        folder_path = file_dir + f'resized_{size[0]}x{size[1]}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for i in trange(len(images), desc=f'[Resize] {size[0]}x{size[1]}'):
            
            img = read_image(images[i])
            img = resizer(img)/255
            save_image(img, folder_path + f'{i}.jpg')
    
    if argv[0] == 'split':
        path = argv[1]
        for p in os.listdir(path):
            break
