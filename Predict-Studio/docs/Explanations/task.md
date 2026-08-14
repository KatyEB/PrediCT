# Phase 1: Implement Harness and Multi-View UI

- `[/]` **Architecture Setup**
  - `[ ]` Add Harness navigation bar to `index.html`.
  - `[ ]` Update `app.js` global state with `activeDirection`.
  - `[ ]` Implement a view router to toggle visibility of different design templates.
- `[ ]` **Implement Views** (using real CT data)
  - `[ ]` **01 Argument**: Build the textual score rationale and highlighted slice exhibits.
  - `[x]` **02 Instrument**: Already built. Need to wrap it in the view router.
  - `[ ]` **03 Contact Sheet**: Build the grid layout of all 120 slices and wire up the selection.
  - `[ ]` **04 Ledger**: Build the dense, scrolling data table view.
  - `[ ]` **05 Field**: Build the coronal projection map view (if applicable/possible with real slices, or map to real lesion coordinates).
- `[ ]` **CSS Styling**
  - `[ ]` Merge the prototype's typography and palette rules into `app.css`.
  - `[ ]` Ensure strict layout rules (no global scroll) are maintained across all views.
- `[ ]` **Final Verification**
  - `[ ]` Test tab switching.
  - `[ ]` Verify real data populates correctly across all active views.
