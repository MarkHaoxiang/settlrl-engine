// A small cream count badge for the corner of a resource / dev-card chip. The
// palette matches the board piles' CountToken, so a hand chip and a table pile
// read the same — and a dark token on a light badge stays legible on any
// terrain fill (where a tinted digit washed out). `value` is usually a count
// but may be any short label (a proven hand range like "0–3", a trade rate).
export default function CountBadge({ value }: { value: React.ReactNode }) {
  return (
    <span
      style={{
        position: "absolute",
        right: -4,
        bottom: -4,
        minWidth: 16,
        height: 16,
        padding: "0 3px",
        borderRadius: 8,
        background: "#FDF6E3",
        border: "1px solid #A08050",
        color: "#2C1A00",
        fontSize: 11,
        fontWeight: 800,
        lineHeight: 1,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        boxShadow: "0 1px 2px rgba(0,0,0,0.35)",
      }}
    >
      {value}
    </span>
  );
}
