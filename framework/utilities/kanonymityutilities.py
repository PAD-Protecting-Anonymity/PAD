import math

class KAnonymityUtilities:
    def can_ensure_k_anonymity(self,anonymity_level,amount_of_inputs):
            if (2*anonymity_level-1)*anonymity_level<amount_of_inputs:
                return True
            return False

    def find_min_amount_of_inputs_for_k(self,anonymity_level):
        amount_of_inputs = (2*anonymity_level-1)*anonymity_level
        return amount_of_inputs

    def find_max_k(self,amount_of_inputs):
        return math.floor((1/4)*(1+math.sqrt(1+8*amount_of_inputs)))