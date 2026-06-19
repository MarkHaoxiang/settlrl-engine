import type { TradeOffer } from "../lib/game";
import { RESOURCE_LABELS, playerName, type ResourceKind } from "../lib/boardData";
import Button from "./Button";
import ResourceGlyph from "./ResourceGlyph";
import s from "./TradeResponsePopover.module.css";

// The incoming-trade prompt: a floating card over the board showing what the
// proposer offers (you get their give, you give their receive) with Accept /
// Reject. Shown to the partner whose turn it is to answer.
export default function TradeResponsePopover({
  offer,
  disabled,
  onAccept,
  onReject,
}: {
  offer: TradeOffer;
  disabled?: boolean;
  onAccept: () => void;
  onReject: () => void;
}) {
  const get = offer.give as ResourceKind;
  const give = offer.receive as ResourceKind;
  return (
    <div className={s.box}>
      <span className={s.header}>{playerName(offer.proposer)} offers you a trade</span>
      <div className={s.deal}>
        <div className={s.side}>
          <span className={s.sideLabel}>You get</span>
          <ResourceGlyph kind={get} px={36} />
          <span className={s.res}>{RESOURCE_LABELS[get]}</span>
        </div>
        <span className={s.arrow}>⇄</span>
        <div className={s.side}>
          <span className={s.sideLabel}>You give</span>
          <ResourceGlyph kind={give} px={36} />
          <span className={s.res}>{RESOURCE_LABELS[give]}</span>
        </div>
      </div>
      <div className={s.actions}>
        <Button disabled={disabled} onClick={onReject}>
          ❌ Reject
        </Button>
        <Button selected disabled={disabled} onClick={onAccept}>
          ✅ Accept
        </Button>
      </div>
    </div>
  );
}
