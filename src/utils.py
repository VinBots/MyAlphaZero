######################################################
#
# AlphaZero algorithm applied to Tic-Tac-Toe
# written by Vincent Manier (vmanier2020@gmail.com)
#
# This module includes some utility functions
#
######################################################
 

class DotDict(dict):
<<<<<<< HEAD
    def __getattr__(self, name):                
=======
    def __getattr__(self, name):
>>>>>>> clean
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)
