import os
import re

import torch


class FileHandler:
    '''
    处理磁盘上的文件操作
    '''

    def __init__(self, folder_path: str):
        # 根据指定的存放模型参数的位置，创建文件夹
        self.folder_path = folder_path
        if not self.folder_path.endswith('/'):
            self.folder_path += '/'
        pass

    def save(self, model, name, folder: str = None):
        '''
        将参数保存在磁盘中

        :param model: model weight and other info binding by user
        :param name: 文件命名
        :param folder: 文件夹路径
        '''

        # 处理文件名称
        illegal_char_re = '[\\\\/*?"<>|]'
        name = re.sub(illegal_char_re, '', name)

        # 处理文件夹名称
        if folder is None:
            save_folder = self.folder_path
        else:
            if not folder.endswith('/'):
                folder += '/'
            save_folder = folder

        os.makedirs(save_folder, exist_ok=True)

        # 保存文件至指定路径
        torch.save(model, save_folder + name)

    def load(self, full_path):
        '''
        从磁盘中加载网络参数等信息

        :param full_path: 要加载的信息路径
        '''
        try:
            model = torch.load(full_path)

        except FileNotFoundError:
            raise FileNotFoundError('File not found on disk: ', full_path)

        # 如果磁盘文件以gpu的方式存储，直接加载到cpu上可能会出现异常，检测到异常时尝试使用另一种方法加载
        except BaseException:
            # load weight saved on gpu device to cpu device
            # see https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/3
            model = torch.load(full_path, map_location=lambda storage, loc: storage)

        return model
