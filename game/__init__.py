import pygame
import game.assets_process

pygame.init()

'''
加载素材
'''
# 加载图片和音效文件
IMAGES, SOUNDS = game.assets_process.load_assets()

'''
设置常量
'''
# 游戏屏幕的尺寸，对应素材中背景图片的宽与高
SCREENWIDTH, SCREENHEIGHT = 288, 512
# 游戏帧率
FPS = 30


'''
游戏设置
'''
# 定义屏幕大小
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
# 定义游戏窗口标题
pygame.display.set_caption('Flappy Bird Demo')
GAMECLOCK = pygame.time.Clock()
