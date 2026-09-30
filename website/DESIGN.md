# BNN design

A clear annotated teaching diagram in a bright office leads into a readable reference.

Independent top navigation and a generous plotting canvas. Source Sans 3 carries prose
and labels; mathematical notation uses a serif fallback. White, near-black, restrained
carmine, and a distinct dark teal for the exact estimator. Colors identify mathematical
roles and are paired with text/line styles.

Core tokens are in public/assets/site.css using OKLCH. No decorative entrance motion.
Animation is explicit, bounded, pausable subsample drawing, with a reduced-motion action
that computes a batch without playing frames. Figures and controls are visible by default.

Layouts are inspected on desktop and narrow phones. Long tables scroll inside labeled
regions. Focus states, native sliders, semantic headings and a mobile menu support keyboard use.
