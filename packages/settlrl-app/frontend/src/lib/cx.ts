// Join class names, dropping falsy entries — for conditional / composed classes
// (`cx(ui.button, selected && ui.selected, className)`).
export const cx = (...classes: (string | false | null | undefined)[]): string =>
  classes.filter(Boolean).join(" ");
