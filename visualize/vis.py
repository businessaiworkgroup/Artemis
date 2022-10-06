# -*- coding: UTF-8 -*-
from turtle import bgcolor, width
from data_structure import ds_21card
import pygame
import sys
import os
import time

bgcolor1 = (46, 139, 87)
font_path  = '../ttf/Minecraft.ttf'
fontcolor = (245, 245 ,245)

def render(state_result, screen_image):
    
    pygame.init()
    # 转化为路径
    banker_cards = StrToImgPath(state_result.banker.card_list_observed)
    # banker牌的数量
    banker_cards_num = len(banker_cards)
    
    # 最小单位的宽度和高度
    block_height = screen_image.get_height() / 20
    block_width = screen_image.get_width() / 20
       
    banker_card_width = 0
     
    players = []
    total_point = []
    for player in state_result.player_list:
        players.append(StrToImgPath(player.card_list_observed))
        total_point.append(player.total_point)
    
    
    # banker
    # 缺少2行牌的情况
    for index, bank_card in enumerate(banker_cards):
        # 加载图像
        banker_card_image = pygame.image.load(bank_card)
        # 调整图像大小
        banker_card_image = pygame.transform.scale(banker_card_image, cal_imageSize(
            block_height, banker_card_image.get_width(), banker_card_image.get_height()))

        banker_card_width = banker_card_image.get_width()

        # 获取位置
        x, y = cal_imagePos(len(banker_cards),index + 1,block_width,block_height,banker_card_image.get_width(),banker_card_image.get_height(),screen_image.get_width(),screen_image.get_height())
        # 图片添加至屏幕
        screen_image.blit(banker_card_image,(x,y))    
        # 添加文字至
        font = pygame.font.Font(font_path , int(block_height))
        # 显示文字
        banker_result = font.render('Banker: %d' %state_result.banker.total_point ,False, fontcolor)
        screen_image.blit(banker_result,(block_width / 2,block_height / 2))   
        pygame.display.flip()
    
    # player
    
    #加载图像

    players_card_images = get_playerCard(players)
    
    # 计算位置的基础信息
    player_num, player_block_width= cal_playerImageInfo(players, screen_image.get_width(), block_height)
    player_block_height = block_height
    card_num = 0

    for index, player_cards in enumerate(players_card_images):
        
        
        for indexs, player_card in enumerate(player_cards):
            # 放缩后图片大小
            player_card_image_width, player_card_image_height=  cal_playerImageSize(
                banker_card_width,player_block_width, player_block_height, player_card.get_width(), player_card.get_height(), screen_image.get_width(), screen_image.get_height())
            
            # 调整图像大小
            player_card = pygame.transform.scale(player_card, (player_card_image_width,player_card_image_height))
            
            # 获取位置
            # x, y = cal_imagePos(len(banker_cards),index + 1,block_width,block_height,banker_card_image.get_width(),banker_card_image.get_height(),screen_image.get_width(),screen_image.get_height())
            
            # 图片添加至屏幕
            start_pos_x = (index+1)*player_block_width + card_num * (player_card_image_width + player_block_width)
            screen_image.blit(player_card,(start_pos_x,screen_image.get_height()/2 +player_block_height * 2))      
            if(indexs == 0):
                #  添加文字至
                font = pygame.font.Font(font_path , int(block_height))
                # 显示文字
                banker_result = font.render('player%d: %d' %(index+1,total_point[index]) ,False, fontcolor)
                # 设置文字位置
                screen_image.blit(banker_result,(start_pos_x,screen_image.get_height()/2)) 
            card_num += 1

        pygame.display.flip()
        time.sleep(2)
    
    
    return 0


def StrToImgPath(decks):
    img_paths = []
    for deck in decks:
        img_path = '../img/' + deck + '.png'
        img_paths.append(img_path)
    return img_paths


def cal_imageSize(block_height, width, height):
    scale = width / height
    height = block_height * 6
    width = scale * height
    return (width, height)


def cal_imagePos(cards_num, card_seq, block_width, block_height, image_width, image_height, screen_width,screen_height):
    width_pos1 = (screen_width - cards_num * image_width - max(0,(cards_num -1))*block_width) /2
    width_pos = width_pos1 + (card_seq-1)*(image_width + block_width)
    # banker
    height_pos = 2 * block_height 
    return width_pos,height_pos

def cal_playerImageInfo(players, screen_weight, block_height):
    player_num = len(players)
    card_num = 0
    for player in players:
        card_num += len(player)
    player_block_width =  screen_weight / (card_num * 2 + ( card_num - 1 ) * 1 + player_num * 1  )
    return player_num, player_block_width


# def cal_playerImagePos(card_num,card_seq, block_width, block_height, image_width, image_height, screen_width,screen_height):
#     width_pos1 = screen_width / crad
#     width_pos = width_pos1 + (card_seq-1)*(image_width + block_width)
#     # banker
#     height_pos = 2 * block_height 
#     return font_pos,font_size,

# 得到所有player的card图像
def get_playerCard(players):
    players_card_images = []
    for player in players:
        player_card_image = []
        for card_img in player:
            card_image = pygame.image.load( card_img)
            player_card_image.append(card_image)
        players_card_images.append(player_card_image)
    
    return players_card_images

#计算player放大缩小的大小
def cal_playerImageSize(banker_card_width,block_width, block_height, width, height,screen_width ,screen_height):
    image_scale_width = min(block_width * 2,banker_card_width)
    image_scale_height = min(image_scale_width * height / width ,(screen_height / 2 - block_height *2))
    return image_scale_width, image_scale_height
    