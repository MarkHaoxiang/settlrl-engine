import type { GameAction } from "../lib/game";
import Button from "./Button";
import s from "./ChoicePopover.module.css";

// A row in the bottom panel listing concrete actions to pick from (resource
// choices: monopoly, year of plenty, maritime trade).
export default function ChoicePopover({
  actions,
  onPick,
  onClose,
}: {
  actions: GameAction[];
  onPick: (flat: number) => void;
  onClose: () => void;
}) {
  return (
    <div className={s.box}>
      <div className={s.header}>
        <span className={s.label}>Choose</span>
        <button className={s.cancel} onClick={onClose}>
          Cancel
        </button>
      </div>
      <div className={s.actions}>
        {actions.map((a) => (
          <Button key={a.flat} onClick={() => onPick(a.flat)}>
            {a.label}
          </Button>
        ))}
      </div>
    </div>
  );
}
