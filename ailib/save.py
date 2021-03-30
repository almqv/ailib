import pickle

def save_network( network, savefile:str ) -> None:
    fh = open(savefile, "wb")
    pickle.dump(network, fh)

    fh.close()
