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

        self.dealer.record_10_cards()  #### added
        self.recorded_card = self.recorded_card + self.dealer.fold_cards
        self.recorded_card = self.recorded_card + self.dealer.used_cards
        # print(self.recorded_card)

        for i in range(2):
            for j in range(self.num_players):
                self.dealer.deal_card(self.players[j])
            self.dealer.deal_card(self.dealer)

        for i in range(self.num_players):
            self.players[i].status, self.players[i].score = self.judger.judge_round(self.players[i])

        self.dealer.status, self.dealer.score = self.judger.judge_round(self.dealer)

        self.winner = {'dealer': [0 for _ in range(self.num_players)]}
        for i in range(self.num_players):
            self.winner['player' + str(i)] = [0 for _ in range(self.num_players)]

        self.player_end = np.array([False for _ in range(self.num_players)])
        self.history = []

        return [self.get_state(game_pointer) for game_pointer in range(self.num_players)]

    def step(self, actions):
        ''' Get the next state

        Args:
            action (str): a specific action of blackjack. (Hit or Stand)

        Returns:/
            dict: next player's state
            int: next plater's id
        '''
        live_player_id = np.where(self.player_end==False)[0]
        hand = [[] for _ in range(self.num_players)]
        next_state = [{} for _ in range(self.num_players)]
        for the_player_id in live_player_id:
            if live_player_id == []:
                break
            if self.allow_step_back:
                p = deepcopy(self.players[the_player_id])
                d = deepcopy(self.dealer)
                w = deepcopy(self.winner)
                self.history.append((d, p, w))

            # print('1',[card.get_index() for card in self.players[the_player_id].hand], actions[the_player_id], the_player_id)
            # Play hit
            if actions[the_player_id] == "hit":
                self.dealer.deal_card(self.players[the_player_id])
                self.players[the_player_id].status, self.players[the_player_id].score = self.judger.judge_round(
                    self.players[the_player_id])
                if self.players[the_player_id].status == 'bust':
                    self.player_end[the_player_id] = True
                    # game over, set up the winner, print out dealer's hand # If bust, pass the game pointer
                    if self.player_end.all():
                        while self.judger.judge_score(self.dealer.hand) < 17:
                            self.dealer.deal_card(self.dealer)
                        self.dealer.status, self.dealer.score = self.judger.judge_round(self.dealer)
                        for i in range(self.num_players):
                            self.judger.judge_game(self, i)
                        self.player_end = np.array([False for _ in range(self.num_players)])


            elif actions[the_player_id] == "stand": # If stand, first try to pass the pointer, if it's the last player, dealer deal for himself, then judge game for everyone using a loop
                self.players[the_player_id].status, self.players[the_player_id].score = self.judger.judge_round(
                    self.players[the_player_id])
                self.player_end[the_player_id] = True
                if self.player_end.all():
                    while self.judger.judge_score(self.dealer.hand) < 17:
                        self.dealer.deal_card(self.dealer)
                    self.dealer.status, self.dealer.score = self.judger.judge_round(self.dealer)
                    for i in range(self.num_players):
                        self.judger.judge_game(self, i)
                    self.player_end = np.array([False for _ in range(self.num_players)])

            # print('2',[card.get_index() for card in self.players[the_player_id].hand], actions[the_player_id], the_player_id)
            # print(self.player_end)
            #

        player_hand = [[] for _ in range(self.num_players)]
        # for the_player_id in range(self.num_players):
        #     next_state['player' + str(the_player_id) + ' hand'] = [[] for _ in range(self.num_players)]

        if self.is_over():
            for the_player_id in range(self.num_players):
                hand[the_player_id]=[card.get_index() for card in self.players[the_player_id].hand]
                dealer_hand = [card.get_index() for card in self.dealer.hand]
                mark = 0
                for i in range(self.num_players-1):
                    if i == the_player_id:
                        mark = 1
                    next_state[the_player_id]['player' + str(i + mark) + ' hand'] = [card.get_index() for card in self.players[i + mark].hand]
                    player_hand[the_player_id].append(next_state[the_player_id]['player' + str(i + mark) + ' hand'])
        else:
            for the_player_id in range(self.num_players):
                mark = 0
                hand[the_player_id] = [card.get_index() for card in self.players[the_player_id].hand]
                for i in range(self.num_players-1):
                    if i == the_player_id:
                        mark = 1
                    next_state[the_player_id]['player' + str(i + mark) + ' hand']=[card.get_index() for card in self.players[i + mark].hand[1:]]
                    player_hand[the_player_id].append(next_state[the_player_id]['player' + str(i + mark) + ' hand'])
                dealer_hand = [card.get_index() for card in self.dealer.hand[1:]]
            # print(player_hand,self.game_pointer)
            # for i in range(self.num_players):
            #     next_state['player' + str(i) + ' hand'] = [card.get_index() for card in self.players[i].hand]
        for the_player_id in range(self.num_players):
            next_state[the_player_id]['dealer hand'] = dealer_hand
            next_state[the_player_id]['actions'] = ('hit', 'stand')
                # print(hand[the_player_id])
                # print(player_hand[the_player_id])
            next_state[the_player_id]['state'] = (hand[the_player_id], player_hand[the_player_id], dealer_hand)
            next_state[the_player_id]['record_card'] = [(hand[the_player_id] + dealer_hand)]
        for i in range(self.num_players):
            if i != the_player_id:
                next_state[the_player_id]['record_card'] = next_state[the_player_id]['record_card'] + next_state[the_player_id]['player' + str(i) + ' hand']
            next_state[the_player_id]['record_card'] = next_state[the_player_id]['record_card'] +self.recorded_card
            # print("2 player_self.recorded_card", self.play_counter%4, self.recorded_card)

        return next_state

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
        return 2

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
        state['actions'] = ('hit', 'stand')
        hand = [card.get_index() for card in self.players[player_id].hand]
        if self.is_over():
            dealer_hand = [card.get_index() for card in self.dealer.hand]
        else:
            dealer_hand = [card.get_index() for card in self.dealer.hand[1:]]
        player_hand = []
        if self.is_over():
            mark = 0
            for i in range(self.num_players-1):
                if i == player_id:
                    mark = 1
                state['player' + str(i + mark) + ' hand'] = [card.get_index() for card in self.players[i + mark].hand]
                player_hand.append(state['player' + str(i + mark) + ' hand'])
            # print('recorded_card:', self.recorded_card)
        else:
            mark = 0
            for i in range(self.num_players-1):
                if i == player_id:
                    mark = 1
                state['player' + str(i + mark) + ' hand'] = [card.get_index() for card in self.players[i + mark].hand[1:]]
                player_hand.append(state['player' + str(i + mark) + ' hand'])

        # for i in range(self.num_players):
        #     state['player' + str(i) + ' hand'] = [card.get_index() for card in self.players[i].hand[1:]] # one hidden card
        state['dealer hand'] = dealer_hand
        state['state'] = (hand, player_hand, dealer_hand)
        # print(player_hand, player_id)
        state['record_card'] = hand + dealer_hand
        for i in range(self.num_players):
            if i != player_id:
                # print(state, player_id)
                state['record_card'] = state['record_card'] + state['player' + str(i) + ' hand']
        state['record_card'] = state['record_card'] + self.recorded_card
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
            if self.winner['player' + str(i)] ==  [0 for _ in range(self.num_players)]:
                return False
        return True
