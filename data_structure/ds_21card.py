# -*- coding: UTF-8 -*-

# 庄家
class Banker:
    # 目前可见的牌
    card_list_observed = []
    # 总点数
    total_point = 0
    # 构造函数
    def __init__(self, card_list_observed, total_point):
        self.card_list_observed = card_list_observed
        self.total_point = total_point

# 玩家
class Player:
    # 目前可见的牌。
    card_list_observed = []
    # 总点数
    total_point = 0
    # 构造函数

    def __init__(self, card_list_observed, total_point):
        self.card_list_observed = card_list_observed
        self.total_point = total_point


# 与可视化界面交互数据结构
class StateResult:
    # 庄家侧
    banker = Banker([], 0)
    # 玩家侧(列表)
    player_list = []
    # 构造函数

    def __init__(self, banker, player_list):
        self.banker = banker
        self.player_list = player_list
