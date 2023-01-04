# This class needs to implement 3 methods:
# - init
# - shutdown
# - get_screen_position

class Eye:
    # Does not take any arguments
    def init(self) -> bool:
        return False

    # Does not take any arguments
    def shutdown(self):
        pass

    # Returns the position on a virtual screen of where the participant is looking  
    # at in normalized 0-1 2D space, with the origin being at the top left
    #
    #   0,0      1,0
    #    ┌────────┐
    #    │        │
    #    │        │
    #    └────────┘
    #   0,1      1,1
    #
    def get_screen_position(self) -> tuple[bool, float, float]:
        return False, 0.0, 0.0