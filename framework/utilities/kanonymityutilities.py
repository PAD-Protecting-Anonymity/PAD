import math

class KAnonymityUtilities:
    def can_ensure_k_anonymity(self,anonymity_level,amount_of_inputs):
            if (2*anonymity_level-1)*5<amount_of_inputs:
                return True
            return False

    def find_min_amount_of_inputs_for_k(self,anonymity_level):
        amount_of_inputs = -5+10*anonymity_level
        return amount_of_inputs

    def find_max_k(self,amount_of_inputs):
        return math.floor((amount_of_inputs+5)/10)