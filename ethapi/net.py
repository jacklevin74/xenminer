from ethapi.base import BaseApi
from ethapi.config import CHAIN_ID


class NetApi(BaseApi):
    def version(self) -> str:
        """
        Returns the current network ID
        """
        self.logger.debug("net_version()")
        return CHAIN_ID
