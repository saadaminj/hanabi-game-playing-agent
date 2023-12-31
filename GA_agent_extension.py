"""Rule-based Chromosome Agent.""" 

from hanabi_learning_environment.rl_env import Agent
import random

def argmax(llist):
    #useful function for arg-max
    return llist.index(max(llist))
    
class RuleAgentChromosome(Agent):
    """Agent that applies a simple heuristic."""

    def __init__(self, config, chromosome=[0,2,5,6], *args, **kwargs): # TODO replace this default chromosome by a good one, plus, possibly, add some new rules into the logic below
        """Initialize the agent."""
        self.config = config
        self.chromosome=chromosome
        assert isinstance(chromosome, list)
        
        # Extract max info tokens or set default to 8.
        self.max_information_tokens = config.get('information_tokens', 8)

    def calculate_all_unseen_cards(self, discard_pile, player_hands, fireworks):
        # All of the cards which we can't see are either in our own hand or in the deck.
        # The other cards must be in the discard pile (all cards of which we have seen and remembered) or in other player's hands.
        colors = ['Y', 'B', 'W', 'R', 'G']
        full_hanabi_deck=[{"color":c, "rank":r} for c in colors for r in [0,0,0,1,1,2,2,3,3,4]]
        assert len(full_hanabi_deck)==50 # full hanabi deck size.

        result=full_hanabi_deck
        # subract off all cards that have been discarded...
        for card in discard_pile:
            if card in result:
                result.remove(card)
        
        # subract off all cards that we can see in the other players' hands...
        for hand in player_hands[1:]:
            for card in hand:
                if card in result:
                    result.remove(card)

        for (color, height) in fireworks.items():
            for rank in range(height):
                card={"color":color, "rank":rank}
                if card in result:
                    result.remove(card)

        # Now we left with only the cards we have never seen before in the game (so these are the cards in the deck UNION our own hand).
        return result             

    def filter_card_list_by_hint(self, card_list, hint):
        # This could be enhanced by using negative hint information, available from observation['pyhanabi'].card_knowledge()[player_offset][card_number]
        filtered_card_list = card_list
        if hint["color"] != None:
            filtered_card_list = [c for c in filtered_card_list if c["color"] == hint["color"]]
        if hint["rank"] != None:
            filtered_card_list = [c for c in filtered_card_list if c["rank"] == hint["rank"]]
        return filtered_card_list


    def filter_card_list_by_playability(self, card_list, fireworks):
        # find out which cards in card list would fit exactly onto next value of its colour's firework
        return [c for c in card_list if self.is_card_playable(c,fireworks)]

    def filter_card_list_by_unplayable(self, card_list, fireworks):
        # find out which cards in card list are always going to be unplayable on its colour's firework
        # This function could be improved by considering that we know a card of value 5 will never be playable if all the 4s for that colour have been discarded.
        return [c for c in card_list if c["rank"]<fireworks[c["color"]]]

    def is_card_playable(self, card, fireworks):
        return card['rank'] == fireworks[card['color']]

    def act(self, observation):
        # this function is called for every player on every turn
        """Act based on an observation."""
        if observation['current_player_offset'] != 0:
            # but only the player with offset 0 is allowed to make an action.  The other players are just observing.
            return None
        
        fireworks = observation['fireworks']
        card_hints=observation['card_knowledge'][0] # This [0] produces the card hints for OUR own hand (player offset 0)
        hand_size=len(card_hints)

        # build some useful lists of information about what we hold in our hand and what team-mates know about their hands.
        all_unseen_cards=self.calculate_all_unseen_cards(observation['discard_pile'],observation['observed_hands'],observation['fireworks'])
        possible_cards_by_hand=[self.filter_card_list_by_hint(all_unseen_cards, h) for h in card_hints]
        playable_cards_by_hand=[self.filter_card_list_by_playability(posscards, fireworks) for posscards in possible_cards_by_hand]
        probability_cards_playable=[len(playable_cards_by_hand[index])/len(possible_cards_by_hand[index]) for index in range(hand_size)]
        useless_cards_by_hand=[self.filter_card_list_by_unplayable(posscards, fireworks) for posscards in possible_cards_by_hand]
        probability_cards_useless=[len(useless_cards_by_hand[index])/len(possible_cards_by_hand[index]) for index in range(hand_size)]
        
        # based on the above calculations, try a sequence of rules in turn and perform the first one that is applicable:
        for rule in self.chromosome:
            if rule in [0,1] and observation['life_tokens']>1:
                threshold=0.8 if rule == 0 else 0.6
                if max(probability_cards_playable)>threshold:
                    card_index=argmax(probability_cards_playable)
                    return {'action_type': 'PLAY', 'card_index': card_index}
                    
            if rule == 2:
                if observation['information_tokens'] > 0:

                    best_prob = 0
                    final_hint = None 

                    # Check if there are any playable cards in the hands of the opponents.
                    for player_offset in range(1, observation['num_players']):
                        player_hand = observation['observed_hands'][player_offset]
                        player_hints = observation['card_knowledge'][player_offset]

                        # Check if the card in the hand of the opponent is playable.
                        for card, hint in zip(player_hand, player_hints):
                            #if card['rank'] == fireworks[card['color']]:
                            if self.is_card_playable(card,fireworks):
                                fireworks = observation['fireworks']
                                player_hand_size=len(player_hints)

                                # build some useful lists of information about what we hold in our hand and what team-mates know about their hands.
                                player_all_unseen_cards=self.calculate_all_unseen_cards(observation['discard_pile'],observation['observed_hands'],observation['fireworks'])
                                
                                if hint['color'] != None:

                                    player_hints.append({'color':card['color'], 'rank':None})
                                    player_possible_cards_by_hand=[self.filter_card_list_by_hint(player_all_unseen_cards, h) for h in player_hints]
                                    player_playable_cards_by_hand=[self.filter_card_list_by_playability(posscards, fireworks) for posscards in player_possible_cards_by_hand]
                                    player_probability_cards_playable=[len(player_playable_cards_by_hand[index])/(len(player_possible_cards_by_hand[index])+0.001) for index in range(player_hand_size+1)]
                                    
                                    if(max(player_probability_cards_playable)>best_prob):
                                        best_prob = max(player_probability_cards_playable)
                                        final_hint = {
                                                'action_type': 'REVEAL_COLOR',
                                                'color': card['color'],
                                                'target_offset': player_offset
                                            }

                                    player_hints.pop(len(player_hints)-1)

                                    
                                if hint['rank'] != None:

                                    player_hints.append({'rank':card['rank'], 'color':None})
                                    player_possible_cards_by_hand=[self.filter_card_list_by_hint(player_all_unseen_cards, h) for h in player_hints]
                                    player_playable_cards_by_hand=[self.filter_card_list_by_playability(posscards, fireworks) for posscards in player_possible_cards_by_hand]
                                    player_probability_cards_playable=[len(player_playable_cards_by_hand[index])/(len(player_possible_cards_by_hand[index])+0.001) for index in range(player_hand_size+1)]
                                    
                                    if(max(player_probability_cards_playable)>best_prob):
                                        best_prob = max(player_probability_cards_playable)
                                        final_hint = {
                                                'action_type': 'REVEAL_RANK',
                                                'rank': card['rank'],
                                                'target_offset': player_offset
                                            }
                                            
                                    player_hints.pop(len(player_hints)-1)
                    
                    if final_hint != None:
                        return final_hint
                                
            if rule == 3 :
                return {'action_type': 'PLAY', 'card_index': argmax(probability_cards_playable)}
            
            if rule == 4:
                threshold=0.4
                if observation['information_tokens'] < self.max_information_tokens:
                    if max(probability_cards_useless)>threshold:
                        card_index=argmax(probability_cards_useless)
                        return {'action_type': 'DISCARD', 'card_index': card_index}
            
            if rule == 5:
                if observation['information_tokens'] < self.max_information_tokens:
                    for card_index, hint in enumerate(card_hints):
                        if hint['color'] is None and hint['rank'] is None:
                            # This card has had hint, so just play it and hope for the best...
                            return {'action_type': 'DISCARD', 'card_index': card_index}
            
            if rule == 6:
                # Discard something
                if observation['information_tokens'] < self.max_information_tokens:
                    return {'action_type': 'DISCARD', 'card_index': 0}# discards the oldest card (card_index 0 will be oldest card)
            
            if rule == 7:
                if observation['information_tokens'] > 0:
                    for i in range(10):
                        player_offset = random.randint(1, observation['num_players']-1)
                        player_hand = observation['observed_hands'][player_offset]
                        player_hints = observation['card_knowledge'][player_offset]
                        # Check if the card in the hand of the opponent is playable.
                        for card, hint in zip(player_hand, player_hints):
                            #if card['rank'] == fireworks[card['color']]:
                            if self.is_card_playable(card,fireworks):
                                if hint['color'] is None:
                                    return {
                                        'action_type': 'REVEAL_COLOR',
                                        'color': card['color'],
                                        'target_offset': player_offset
                                    }
                                elif hint['rank'] is None:
                                    return {
                                        'action_type': 'REVEAL_RANK',
                                        'rank': card['rank'],
                                        'target_offset': player_offset
                                    }
            if rule == 8:
                # Check if it's possible to hint a card to your colleagues.
                if observation['information_tokens'] > 0:
                    # Check if there are any playable cards in the hands of the opponents.
                    for player_offset in range(1, observation['num_players']):
                        player_hand = observation['observed_hands'][player_offset]
                        player_hints = observation['card_knowledge'][player_offset]
                        # Check if the card in the hand of the opponent is playable.
                        for card, hint in zip(player_hand, player_hints):
                            #if card['rank'] == fireworks[card['color']]:
                            if self.is_card_playable(card,fireworks):
                                if hint['color'] is None:
                                    return {
                                        'action_type': 'REVEAL_COLOR',
                                        'color': card['color'],
                                        'target_offset': player_offset
                                    }
                                elif hint['rank'] is None:
                                    return {
                                        'action_type': 'REVEAL_RANK',
                                        'rank': card['rank'],
                                        'target_offset': player_offset
                                    }

            
        
        # The chromosome needs to be defined so the program never gets to here.  
        # E.g. always include rules 5 and 6 in the chromosome somewhere to ensure this never happens..        
        
        if observation['information_tokens'] < self.max_information_tokens:
            return {'action_type': 'DISCARD', 'card_index': 0}# discards the oldest card (card_index 0 will be oldest card)
        return {'action_type': 'PLAY', 'card_index': argmax(probability_cards_playable)}
        raise Exception("No rule fired for game situation - faulty rule set")
