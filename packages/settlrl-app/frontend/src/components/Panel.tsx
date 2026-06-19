import { cx } from "../lib/cx";
import ui from "../styles/ui.module.css";

// The standard panel surface (paper/glass over the tokens). Per-instance layout
// (padding, min-width, radius overrides) composes through `style`/`className`.
export default function Panel({
  className,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cx(ui.panel, className)} {...rest} />;
}
