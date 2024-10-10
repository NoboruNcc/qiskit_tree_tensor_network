import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SimpleToTensor:
    """
    入力データをPyTorchのテンソルに変換するシンプルな変換クラス.

    Attributes:
        dtype: 変換後のテンソルのデータ型

    Methods:
        __call__: 入力をテンソルに変換する
        __repr__: クラスの文字列表現を返す
    """

    def __init__(self, dtype=int):
        self.dtype = dtype

    def __call__(self, x):
        return torch.tensor(x, dtype=self.dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

def transform_label(label):
    """
    ラベルを1と-1に変換する関数.

    Args:
        label: 入力ラベル

    Returns:
        int: 1（入力が1の場合）または-1（それ以外の場合）
    """
    return 1 if label == 1 else -1

target_transform = transforms.Compose([
    SimpleToTensor(),
    transform_label
])


class TransformableDataset(Dataset):
    """
    変換可能なデータセットクラス.
    データとターゲットに対して変換を適用できるカスタムデータセット.

    Attributes:
        data: 元のデータ
        target: 対応するターゲット（ラベル）
        transform: データに適用する変換（オプション）
        target_transform: ターゲットに適用する変換（オプション）

    Methods:
        __len__: データセットの長さを返す
        __getitem__: 指定されたインデックスのデータとターゲットのペアを返す
    """

    def __init__(self, data, target, transform=None, target_transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        
        y = self.target[index]
        if self.target_transform:
            y = self.target_transform(y)
            
        return x, y
