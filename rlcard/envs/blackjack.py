import numpy as np
from collections import OrderedDict

from rlcard.envs import Env
from rlcard.games.blackjack import Game

DEFAULT_GAME_CONFIG = {
        'game_num_players': 1,
        'game_num_decks': 1
        }

class BlackjackEnv(Env):
    ''' Blackjack Environment
    '''

    def __init__(self, config):
        ''' Initialize the Blackjack environment
        '''
        self.name = 'blackjack'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        if self.num_actions == 3:
            self.actions = ['hit', 'stand', 'prior_hit']
        elif self.num_actions == 2:
            self.actions = ['hit', 'stand']

        if self.share_policy and self.num_players>1:
            self.state_shape = [[2+2] for _ in range(self.num_players)] # 2->3
        elif self.num_players>1:
            self.state_shape = [[2 + 1] for _ in range(self.num_players)]  # 2->3
        else:
            self.state_shape = [[2+ 10] for _ in range(self.num_players)]  # 2->3
        self.action_shape = [None for _ in range(self.num_players)]

    def _get_legal_actions(self):
        ''' Get all leagal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        encoded_action_list = []
        for i in range(len(self.actions)):
            encoded_action_list.append(i)
        return encoded_action_list

    def _extract_state(self, state):
        ''' Extract the state representation from state dictionary for agent

        Args:
            state (dict): Original state from the game

        Returns:
            observation (list): combine the player's score and dealer's observable score for observation
        '''
        cards = state['state']
        my_cards = cards[0]
        dealer_cards = cards[1]

        card_mean = 6.923
        remained_card_value = 0
        normalisation = False
        card_distribution = [0 for _ in range(10)]
        full_card_distribution = [0 for _ in range(13)]
        for the_card in state['record_card']:
            if the_card[1].isdigit():
                card_distribution[int(the_card[1])-1] +=1
                full_card_distribution[int(the_card[1]) - 1] += 1
            elif the_card[1] == 'A':
                card_distribution[0] += 1
                full_card_distribution[0] += 1
            else:
                card_distribution[9] += 1
                if the_card[1] == 'T':
                    full_card_distribution[9] += 1
                elif the_card[1] == 'J':
                    full_card_distribution[10] += 1
                elif the_card[1] == 'Q':
                    full_card_distribution[11] += 1
                elif the_card[1] == 'K':
                    full_card_distribution[12] += 1
            if normalisation:
                if the_card[1].isdigit():
                    remained_card_value = remained_card_value + (int(the_card[1]) - card_mean) / 5 / self.game.num_decks
                elif the_card[1] == 'A':
                    remained_card_value = remained_card_value + (6 - card_mean) / 5 / self.game.num_decks
                else:
                    remained_card_value = remained_card_value + (10 - card_mean) / 5 / self.game.num_decks
            else:
                if the_card[1].isdigit():
                    if int(the_card[1]) <7:
                        remained_card_value -= 1/ self.game.num_decks
                else:
                    remained_card_value += 1/ self.game.num_decks

        noise = np.random.normal(remained_card_value/5, 0.5, 1)

        my_score = get_score(my_cards)
        dealer_score = get_score(dealer_cards)


        # obs = np.array([my_score, dealer_score, remained_card_value])
        obs = np.array([my_score/31, dealer_score/31] + [card/8 for card in card_distribution])
        # obs = np.array([my_score, dealer_score] + [card for card in full_card_distribution])
        # obs = np.array([my_score / 31, dealer_score / 31, remained_card_value / 10 + 0.5])
        # print('player_id',state['player_id'] , self.game.game_pointer)
        # if self.num_players > 1:
        #     obs = np.array([my_score / 31, dealer_score / 31, remained_card_value / 10 + 0.5])
        # else:
        #     obs = np.array([my_score/31, dealer_score/31])
        if self.share_policy and self.num_players>1:
            obs = np.append(obs, state['player_id']/(self.num_players-1))
        # print('observation',obs)
        # print(self.game.is_over(),'player' + str(self.game.game_pointer) + ' hand', state['player' + str(self.game.game_pointer) + ' hand'])
        # print(self.game.is_over(),'recorded_card',self.game.game_pointer,state['record_card'])



        legal_actions = OrderedDict({i: None for i in range(len(self.actions))})
        extracted_state = {'obs': obs, 'legal_actions': legal_actions}
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in self.actions]
        extracted_state['action_record'] = self.action_recorder

        extracted_state['remained_card_value'] = remained_card_value
        # print(extracted_state['legal_actions'])
        return extracted_state

    def get_payoffs(self):
        ''' Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        '''
        payoffs = []

        for i in range(self.num_players):
            if self.game.winner['player' + str(i)] == 2:
                payoffs.append(1)  # Dealer bust or player get higher score than dealer
            elif self.game.winner['player' + str(i)] == 1:
                payoffs.append(0)  # Dealer and player tie
            else:
                payoffs.append(-1)  # Player bust or Dealer get higher score than player

        return np.array(payoffs)


    def _decode_action(self, action_id):
        ''' Decode the action for applying to the game

        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        '''
        return self.actions[action_id]

rank2score = {"A":11, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "T":10, "J":10, "Q":10, "K":10}
def get_score(hand):
    score = 0
    count_a = 0
    for card in hand:
        score += rank2score[card[1:]]
        if card[1] == 'A':
            count_a += 1
    while score > 21 and count_a > 0:
        count_a -= 1
        score -= 10
    return score
