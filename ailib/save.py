import pickle


def save_network(network, savefile: str) -> None:
    fh = open(savefile, "wb")
    pickle.dump(network, fh)

    fh.close()


def load_network(network, savefile: str):
    fh = open(savefile, "rb")
    network = pickle.load(fh)
    fh.close()

    return network
