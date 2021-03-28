import pickle

def save_network( network, savefile:str ) -> None:
    fh = open(savefile, "w")
    pickle.dump(network, fh)
