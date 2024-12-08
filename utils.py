import torch

def batch(states):
    """
    Turns a list of states into a single tensor for fast processing.
    This function also chunks (splits) each state into a (h, c) pair"""
    return torch.cat(states, 0).chunk(2, 1)

def unbatch(state):
    """
    Turns a tensor back into a list of states.
    First, (h, c) are merged into a single state.
    Then the result is split into a list of sentences.
    """
    return torch.split(torch.cat(state, 1), 1, 0)