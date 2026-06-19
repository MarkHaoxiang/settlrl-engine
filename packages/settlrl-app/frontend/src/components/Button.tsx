import { cx } from "../lib/cx";
import ui from "../styles/ui.module.css";

type Variant = "default" | "small";

// The standard button: tokenised look-and-feel from ui.module.css. `selected`
// adds the accent wash; `style`/`className` still compose for per-instance tweaks
// (a dynamic width, an extra layout class).
export default function Button({
  variant = "default",
  selected = false,
  className,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  selected?: boolean;
}) {
  const cls = cx(
    variant === "small" ? ui.buttonSmall : ui.button,
    selected && ui.selected,
    className
  );
  return <button className={cls} {...rest} />;
}
