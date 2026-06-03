"""Tourguide Python SDK.

Drive a running Tourguide visual workspace from a script or notebook. Calls
the same HTTP Workspace API (via the local bridge) that the MCP adapter uses.

    from tourguide_client import TourguideSession

    s = TourguideSession.attach()
    s.get_session()
    s.show_plot(code="plt.hist(df_mitochondria['volume_nm_3'])", title="Volumes")
    s.save_session_state("interesting state")
"""

from .session import TourguideSession
from .schemas import WORKSPACE_OPS, WorkspaceError

__all__ = ["TourguideSession", "WORKSPACE_OPS", "WorkspaceError"]
__version__ = "0.1.0"
