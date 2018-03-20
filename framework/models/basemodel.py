class BaseModel:
    def __init__(self,data):
        self.data = data # dataset where each row corresponds to an individual
        self.datasize = data.shape[0] # the number of indivduals contained in the database
        self._groups = []    

    def find_clusters(self):
        raise NotImplementedError('users must define find_clusters in class to use this base class')

    def get_groups(self):
        return self._groups
        
