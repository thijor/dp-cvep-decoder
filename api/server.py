from pathlib import Path

from dareplane_utils.default_server.server import DefaultServer
from fire import Fire

from cvep_decoder.online_decoding import online_decoder_factory
from cvep_decoder.utils.logging import logger


def main(
    port: int = 8080,
    ip: str = "127.0.0.1",
    loglevel: int = 10,
    conf_pth: Path = Path("./configs/decoder.toml"),
):
    logger.setLevel(loglevel)

    logger.debug(f"Initializing decoder with {conf_pth=}")
    decoder = online_decoder_factory(conf_pth)

    pcommand_map = {
        # "CREATE CLASSIFIER": partial(
        #     create_classifier, subject="P001", session="S001", run="001"
        # ),
        "CONNECT_DECODER": decoder.init_all,
        "DECODE ONLINE": decoder.run,
    }

    server = DefaultServer(
        port, ip=ip, pcommand_map=pcommand_map, name="cvep_decoder_server"
    )

    # initialize to start the socket
    server.init_server()
    # start processing of the server
    server.start_listening()

    return 0


if __name__ == "__main__":
    Fire(main)
