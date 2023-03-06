from copy import deepcopy
import numpy as np

from rlcard.games.blackjack import Dealer
from rlcard.games.blackjack import Player
from rlcard.games.blackjack import Judger

class BlackjackGame:

    def __init__(self, allow_step_back=False):
        ''' Initialize the class Blackjack Game
        '''
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.play_counter = 0
        self.num_of_turns = 1

    def configure(self, game_config):
        ''' Specifiy some game specific parameters, such as number of players
        '''
        self.num_players = game_config['game_num_players']
        self.num_decks = game_config['game_num_decks']

    def init_game(self):
        ''' Initialilze the game

        Returns:
            state (dict): the first state of the game
            player_id (int): current player's id
        '''
        if self.play_counter > 99:
            self.play_counter = 0

        if self.play_counter%self.num_of_turns==0:
            self.recorded_card = []
        self.play_counter += 1

        self.dealer = Dealer(self.np_random,self.play_counter, self.num_decks, self.num_of_turns)

        self.players = []
        for i in range(self.num_players):
            self.players.append(Player(i, self.np_random))

        self.judger = Judger(self.np_random)

        # self.dealer.record_10_cards()  #### added
        # print(self.dealer.fold_cards)
        # self.recorded_card = self.recorded_card + self.dealer.fold_cards
        self.recorded_card = self.recorded_card + self.dealer.used_cards

        for i in range(2):
            for j in range(self.num_players):
                self.dealer.deal_card(self.players[j])
            self.dealer.deal_card(self.dealer)

        for i in range(self.num_players):
            self.players[i].status, self.players[i].score = self.judger.judge_round(self.players[i])

        self.dealer.status, self.dealer.score = self.judger.judge_round(self.dealer)

        self.winner = {'dealer': 0}
        for i in range(self.num_players):
            self.winner['player' + str(i)] = 0

        self.history = []
        self.game_pointer = 0
        self.hand_pointer = np.array([False for _ in range(self.num_players)])

        return self.get_state(self.game_pointer), self.game_pointer

    def decide_prior(self, actions):
        for id in range(self.num_players):
            if actions[id]==2 and self.hand_pointer[id] == False:
                self.game_pointer = id
                return id
        id = self.game_pointer
        return id


    def step(self, action):
        ''' Get the next state

        Args:
            Added action (str): a specific action of blackjack. (Hand)

            action (str): a specific action of blackjack. (Hit or Stand)

        Returns:/
            dict: next player's state
            int: next plater's id
        '''
        if self.allow_step_back:
            p = deepcopy(self.players[self.game_pointer])
            d = deepcopy(self.dealer)
            w = deepcopy(self.winner)
            self.history.append((d, p, w))

        next_state = {}
        # if action == 'hand':
        # Play hit
        if action != "stand": # "hit" or 'prior_hit':
            self.dealer.deal_card(self.players[self.game_pointer])
            self.players[self.game_pointer].status, self.players[self.game_pointer].score = self.judger.judge_round(
                self.players[self.game_pointer])

            if self.players[self.game_pointer].status == 'bust':
                self.hand_pointer[self.game_pointer] = True
                # game over, set up the winner, print out dealer's hand # If bust, pass the game pointer
                # if self.game_pointer >= self.num_players - 1:  ##### change needed
                if self.hand_pointer.all():  ##### change needed
                    while self.judger.judge_score(self.dealer.hand) < 17:
                        self.dealer.deal_card(self.dealer)
                    self.dealer.status, self.dealer.score = self.judger.judge_round(self.dealer)
                    for i in range(self.num_players):
                        self.judger.judge_game(self, i) 
                    self.game_pointer = 0
                else:
                    for id in range(self.num_players):
                        if self.hand_pointer[id] == False:
                            self.game_pointer = id
                    # self.game_pointer += 1
                
        elif action == "stand": # If stand, first try to pass the pointer, if it's the last player, dealer deal for himself, then judge game for everyone using a loop
            self.players[self.game_pointer].status, self.players[self.game_pointer].score = self.judger.judge_round(
                self.players[self.game_pointer])
            self.hand_pointer[self.game_pointer] = True
            if self.hand_pointer.all():  ##### change needed
            # if self.game_pointer >= self.num_players - 1:
                while self.judger.judge_score(self.dealer.hand) < 17:
                    self.dealer.deal_card(self.dealer)
                self.dealer.status, self.dealer.score = self.judger.judge_round(self.dealer)
                for i in range(self.num_players):
                    self.judger.judge_game(self, i)
                self.game_pointer = 0
            else:
                for i in range(self.num_players):
                    if self.hand_pointer[i] == False:
                        self.game_pointer = i
                    # self.game_pointer += 1

        # print(self.game_pointer)
        hand = [card.get_index() for card in self.players[self.game_pointer].hand]

        if self.is_over():
            dealer_hand = [card.get_index() for card in self.dealer.hand]
            # self.recorded_card = self.recorded_card + self.dealer.used_cards
            # print('recorded_card:', self.recorded_card)
        else:
            dealer_hand = [card.get_index() for card in self.dealer.hand[1:]]

        for i in range(self.num_players):
            next_state['player' + str(i) + ' hand'] = [card.get_index() for card in self.players[i].hand]
        next_state['dealer hand'] = dealer_hand
        next_state['actions'] = ('hit', 'stand', 'prior_hit')
        next_state['state'] = (hand, dealer_hand)
        next_state['player_id'] = self.game_pointer
        next_state['record_card'] = hand  + dealer_hand
        # next_state['record_card'] = ['player' + str(self.game_pointer) + ' hand'] + hand + ['dealer']+dealer_hand
        for i in range(self.num_players):
            if i != self.game_pointer:
                next_state['record_card'] = next_state['record_card']+ next_state['player' + str(i) + ' hand'][1:]
                # next_state['record_card'] = next_state['record_card'] + ['player' + str(i) + ' hand'] + next_state['player' + str(i) + ' hand'][1:]
        next_state['record_card'] = next_state['record_card'] + self.recorded_card
        # next_state['record_card'] = next_state['record_card'] + ['old_cards:'] +self.recorded_card
        # print("2 player_self.recorded_card", self.play_counter%4, self.recorded_card)
        return next_state, self.game_pointer

    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            Status (bool): check if the step back is success or not
        '''
        #while len(self.history) > 0:
        if len(self.history) > 0:
            self.dealer, self.players[self.game_pointer], self.winner = self.history.pop()
            return True
        return False

    def get_num_players(self):
        ''' Return the number of players in blackjack

        Returns:
            number_of_player (int): blackjack only have 1 player
        '''
        return self.num_players

    @staticmethod
    def get_num_actions():
        ''' Return the number of applicable actions

        Returns:
            number_of_actions (int): there are only two actions (hit and stand)
        '''
        return 2 + 1

    def get_player_id(self):
        ''' Return the current player's id

        Returns:
            player_id (int): current player's id
        '''
        return self.game_pointer

    def get_state(self, player_id):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            state (dict): corresponding player's state
        '''
        '''
                before change state only have two keys (action, state)
                but now have more than 4 keys (action, state, player0 hand, player1 hand, ... , dealer hand)
                Although key 'state' have duplicated information with key 'player hand' and 'dealer hand', I couldn't remove it because of other codes
                To remove it, we need to change dqn agent too in my opinion
                '''
        state = {}
        state['actions'] = ('hit', 'stand', 'prior_hit')
        hand = [card.get_index() for card in self.players[player_id].hand]
        if self.is_over():
            dealer_hand = [card.get_index() for card in self.dealer.hand]
            # self.recorded_card = self.recorded_card + self.dealer.used_cards
            # print('recorded_card:', self.recorded_card)
        else:
            dealer_hand = [card.get_index() for card in self.dealer.hand[1:]]

        for i in range(self.num_players):
            state['player' + str(i) + ' hand'] = [card.get_index() for card in self.players[i].hand[1:]] # one hidden card
        state['dealer hand'] = dealer_hand
        state['state'] = (hand, dealer_hand)
        state['player_id'] = player_id
        state['record_card'] = hand + dealer_hand
        for i in range(self.num_players):
            if i != player_id:
                state['record_card'] = state['record_card'] + state['player' + str(i) + ' hand']
        # state['record_card'] = state['record_card'] + self.recorded_card
        # print("1 player_self.recorded_card",  self.play_counter%4,self.recorded_card)
        return state

    def is_over(self):
        ''' Check if the game is over

        Returns:
            status (bool): True/False
        '''
        '''
                I should change here because judger and self.winner is changed too
                '''
        for i in range(self.num_players):
            if self.winner['player' + str(i)] == 0:
                return False
        return True
