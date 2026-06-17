// Simple monochrome line icons. They draw with `currentColor`, so they inherit
// the surrounding text colour and follow the theme / selected state.

function Glyph({ size = 15, children }: { size?: number; children: React.ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ flexShrink: 0 }}
      aria-hidden
    >
      {children}
    </svg>
  );
}

// A person: a human-controlled seat.
export const HumanIcon = ({ size }: { size?: number }) => (
  <Glyph size={size}>
    <circle cx="12" cy="8" r="3.5" />
    <path d="M5 20c0-3.9 3.1-7 7-7s7 3.1 7 7" />
  </Glyph>
);

// A robot head: a bot-controlled seat.
export const BotIcon = ({ size }: { size?: number }) => (
  <Glyph size={size}>
    <rect x="5" y="9" width="14" height="10" rx="2.5" />
    <path d="M12 9V5.5" />
    <circle cx="12" cy="4" r="1" />
    <circle cx="9.5" cy="14" r="1" fill="currentColor" stroke="none" />
    <circle cx="14.5" cy="14" r="1" fill="currentColor" stroke="none" />
  </Glyph>
);

// A hex tile: the board / map.
export const MapIcon = ({ size }: { size?: number }) => (
  <Glyph size={size}>
    <path d="M12 3 19 7.5 19 16.5 12 21 5 16.5 5 7.5Z" />
  </Glyph>
);
