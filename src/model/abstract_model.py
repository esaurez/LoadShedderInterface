from abc import ABC, abstractmethod

class UtilityModel(ABC):
    
    # TODO Do we really need this method and not a constructor ?
    #@abstractmethod
    #def init_shedding(self, model_path):
    #    pass

    @abstractmethod
    def get_utility(self, features):
        pass

    @abstractmethod
    def get_utility_threshold(self, drop_rate):
        pass
