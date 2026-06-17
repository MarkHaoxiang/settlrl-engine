"""Per-seat snapshot views: what one requester is allowed to see.

The redaction boundary for hidden information lives here, and only here:
routes never touch a session's state except through :func:`game_model`. The
event stream pushes the same per-seat snapshot.
"""

from settlrl_game.actions import decode_actions
from settlrl_game.convert import board_to_model
from settlrl_game.models import GameModel
from settlrl_render.game.games import GameHandle


def game_model(handle: GameHandle, owned: set[int]) -> GameModel:
    """The snapshot as one requester sees it (caller holds the game's lock).

    ``owned`` is the requester's proven seats: it decides ``your_turn``, which
    legal actions ship, whose hands stay unredacted, and the belief observer.
    Spectators (no seats) get the public view: counts, board, log — no hands.
    Once the game is over every hand is revealed, to anyone: there is no longer
    a position to protect, and the final standings show the full breakdown.
    """
    session = handle.session
    status = session.status()
    # A game waiting in its lobby (unclaimed human seats) serves no actions and
    # nobody's turn is live, so play can't begin before every player is in.
    status.your_turn = handle.ready() and (not status.terminal) and status.acting_player in owned
    actions = decode_actions(session.legal_flat()) if status.your_turn else []
    observer = (
        status.acting_player
        if status.acting_player in owned
        else min(owned)
        if owned
        else None
    )
    board = board_to_model(session.game)
    if not status.terminal:
        for player in board.players:
            if player.player not in owned:
                player.resources = None
                player.dev_card_types = None
    return GameModel(
        id=handle.id,
        version=handle.version,
        board=board,
        status=status,
        actions=actions,
        bot_move=handle.bot_move,
        log=session.log(),
        belief=session.belief(observer) if observer is not None else None,
        seats_claimed=sorted(handle.claims),
        your_seats=sorted(owned),
    )
