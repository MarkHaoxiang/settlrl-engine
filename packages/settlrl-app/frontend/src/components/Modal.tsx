import * as Dialog from "@radix-ui/react-dialog";
import s from "./Modal.module.css";

// A modal dialog on Radix: a backdrop plus focus-trapped content that an outside
// press or Escape dismisses (calling onClose). The visible card is the caller's
// children; `title` names the dialog for assistive tech (rendered off-screen).
// `onEscapeKeyDown` lets a multi-step dialog intercept Escape — preventDefault to
// back out a sub-page instead of closing.
export default function Modal({
  open = true,
  onClose,
  title,
  onEscapeKeyDown,
  children,
}: {
  open?: boolean;
  onClose: () => void;
  title: string;
  onEscapeKeyDown?: (e: KeyboardEvent) => void;
  children: React.ReactNode;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={(o) => !o && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className={s.overlay} />
        <Dialog.Content className={s.content} onEscapeKeyDown={onEscapeKeyDown} aria-describedby={undefined}>
          <Dialog.Title className={s.srOnly}>{title}</Dialog.Title>
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
