from fire import Fire

from decoder.decoder import create_classifier, decode
from decoder.utils.logging import logger
from functools import partial

from dareplane_utils.default_server.server import DefaultServer


def main(port: int = 8080, ip: str = "127.0.0.1", loglevel: int = 10):
    logger.setLevel(loglevel)

    pcommand_map = {
        "CREATE CLASSIFIER": partial(create_classifier, subject="P001", session="S001", run="001"),
        "DECODE ONLINE": partial(decode, subject="P001", session="S001")
    }

    server = DefaultServer(
        port, ip=ip, pcommand_map=pcommand_map, name="decoder_server"
    )

    # initialize to start the socket
    server.init_server()
    # start processing of the server
    server.start_listening()

    return 0


if __name__ == "__main__":
    Fire(main)
