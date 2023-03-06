from rlcard.utils import init_standard_deck
import numpy as np

class BlackjackDealer:

    def __init__(self, np_random, play_num, num_decks, num_of_turns):
        ''' Initialize a Blackjack dealer class
        '''
        self.np_random = np_random
        self.num_decks = num_decks
        self.num_of_turns = num_of_turns
        self.deck = init_standard_deck()
        if self.num_decks not in [0, 1]:  # 0 indicates infinite decks of cards
            self.deck = self.deck * self.num_decks  # copy m standard decks of cards
        if play_num %self.num_of_turns == 0:
            self.shuffle()
        self.hand = []
        self.status = 'alive'
        self.score = 0
        self.used_cards = []
        self.fold_cards = []

    def shuffle(self):
        ''' Shuffle the deck
        '''
        shuffle_deck = np.array(self.deck)
        self.np_random.shuffle(shuffle_deck)
        self.deck = list(shuffle_deck)
        self.used_cards = []
        self.fold_cards = []

    def record_card(self, card):
        self.used_cards.append(card.get_index())

    def deal_card(self, player):
        ''' Distribute one card to the player

        Args:
            player_id (int): the target player's id
        '''
        idx = self.np_random.choice(len(self.deck))
        card = self.deck[idx]
        if self.num_decks != 0:  # If infinite decks, do not pop card from deck
            self.deck.pop(idx)
        self.record_card(card)
        # card = self.deck.pop()
        player.hand.append(card)

    def record_10_cards(self):
        for _ in range(0):
            idx = self.np_random.choice(len(self.deck))
            card = self.deck[idx]
            if self.num_decks != 0:  # If infinite decks, do not pop card from deck
                self.deck.pop(idx)
            self.fold_cards.append(card.get_index())
        # self.record_card(card)

