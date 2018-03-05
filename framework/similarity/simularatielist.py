class SimularatieList:
    def __init__(self):
        self.simularaties = []
    
    def get_distance(self,data):
        distance = None
        for simularatie in self.simularaties:
            tempdistance =simularatie.get_distance(data)
            if distance is not None:
                distance['distance'] = distance['distance'] + tempdistance['distance']
            else:
                distance = tempdistance
        distance = distance.sort_values('distance')
        return distance

    def add_simularatie(self,simularatie):
        self.simularaties.append(simularatie)