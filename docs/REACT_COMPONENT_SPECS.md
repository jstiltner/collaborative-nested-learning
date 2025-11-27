
# React Component Specifications for jasonstiltner.com/nested-learning

This document provides detailed specifications and implementation prompts for building the interactive visualization components.

---

## Component 1: ForgettingHeatmap

### Purpose
Visualize how accuracy on previously learned tasks degrades as new tasks are learned. This is the "hero" visualization that immediately communicates the catastrophic forgetting problem.

### Data Interface

```typescript
interface ForgettingHeatmapProps {
  data: {
    tasks: string[];  // ['Task 1', 'Task 2', ...]
    accuracyMatrix: (number | null)[][];  // 5x5 matrix, null for future tasks
    configuration: string;  // '3-level', '5-level', '7-level'
    bridgesEnabled: boolean;
  };
  width?: number;
  height?: number;
  animated?: boolean;
}
```

### Sample Data

```javascript
const sampleData = {
  tasks: ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5'],
  accuracyMatrix: [
    [98.5, null, null, null, null],
    [35.8, 100.0, null, null, null],
    [17.3, 17.2, 96.4, null, null],
    [16.5, 13.5, 11.8, 37.9, null],
    [15.9, 15.0, 11.7, 10.1, 36.2],
  ],
  configuration: '3-level',
  bridgesEnabled: false,
};
```

### Visual Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Forgetting Heatmap                        │
│                                                              │
│         Task 1   Task 2   Task 3   Task 4   Task 5          │
│        ┌───────┬───────┬───────┬───────┬───────┐            │
│ After  │ 98.5% │       │       │       │       │ Task 1     │
│ Train  ├───────┼───────┼───────┼───────┼───────┤            │
│        │ 35.8% │100.0% │       │       │       │ Task 2     │
│        ├───────┼───────┼───────┼───────┼───────┤            │
│        │ 17.3% │ 17.2% │ 96.4% │       │       │ Task 3     │
│        ├───────┼───────┼───────┼───────┼───────┤            │
│        │ 16.5% │ 13.5% │ 11.8% │ 37.9% │       │ Task 4     │
│        ├───────┼───────┼───────┼───────┼───────┤            │
│        │ 15.9% │ 15.0% │ 11.7% │ 10.1% │ 36.2% │ Task 5     │
│        └───────┴───────┴───────┴───────┴───────┘            │
│                                                              │
│  ████ >80%   ████ 40-80%   ████ <40%   ░░░░ Not yet trained │
└─────────────────────────────────────────────────────────────┘
```

### Color Scheme

```javascript
const colorScale = {
  high: '#22c55e',      // Green for >80%
  medium: '#eab308',    // Yellow for 40-80%
  low: '#ef4444',       // Red for <40%
  null: '#f3f4f6',      // Light gray for not yet trained
  diagonal: '#3b82f6',  // Blue highlight for diagonal (peak accuracy)
};
```

### Implementation Prompt for React Developer

```
Create a React component called ForgettingHeatmap using D3.js for the visualization.

Requirements:
1. Render a 5x5 grid where each cell represents accuracy on a task after training on another task
2. Use a diverging color scale: green (#22c55e) for >80%, yellow (#eab308) for 40-80%, red (#ef4444) for <40%
3. Null values (future tasks) should be light gray (#f3f4f6)
4. The diagonal should have a subtle blue border to highlight peak accuracy
5. On hover, show a tooltip with:
   - "After training on Task X, accuracy on Task Y: Z%"
   - If on diagonal: "Peak accuracy (just learned)"
   - If below diagonal: "Forgetting: -X% from peak"
6. Add smooth CSS transitions for hover states
7. If animated prop is true, reveal cells row by row with a 500ms delay between rows
8. Include axis labels: X-axis "Task Evaluated", Y-axis "After Training On"
9. Add a color legend below the heatmap
10. Calculate and display "Average Forgetting: X%" below the legend

Use Tailwind CSS for layout and styling. The component should be responsive and work on mobile.

Bonus: Add a "Compare" mode that shows two heatmaps side by side (bridges on vs off).
```

---

## Component 2: DepthTradeoffScatter

### Purpose
Show the trade-off between architecture depth, accuracy, and forgetting. Help users understand that there's no free lunch—deeper hierarchies reduce forgetting but make learning harder.

### Data Interface

```typescript
interface DepthTradeoffScatterProps {
  data: {
    levels: number;
    accuracy: number;
    forgetting: number;
    bridges: boolean;
    label: string;
  }[];
  highlightBest?: boolean;
  showParetoFrontier?: boolean;
}
```

### Sample Data

```javascript
const sampleData = [
  { levels: 3, accuracy: 17.78, forgetting: 70.03, bridges: false, label: '3-level (no bridges)' },
  { levels: 3, accuracy: 15.50, forgetting: 64.72, bridges: true, label: '3-level (bridges)' },
  { levels: 5, accuracy: 16.34, forgetting: 69.92, bridges: false, label: '5-level (no bridges)' },
  { levels: 5, accuracy: 18.54, forgetting: 73.83, bridges: true, label: '5-level (bridges)' },
  { levels: 7, accuracy: 13.64, forgetting: 56.85, bridges: false, label: '7-level (no bridges)' },
  { levels: 7, accuracy: 12.94, forgetting: 56.50, bridges: true, label: '7-level (bridges)' },
];
```

### Visual Design

```
┌─────────────────────────────────────────────────────────────┐
│                  Accuracy vs. Forgetting                     │
│                                                              │
│  20% ┤                    ◉ 5-level (bridges)               │
│      │                                                       │
│  18% ┤              ○ 3-level                               │
│      │                  ○ 5-level                            │
│  16% ┤                                                       │
│      │          ◉ 3-level (bridges)                         │
│  14% ┤                                                       │
│      │                              ○ 7-level                │
│  12% ┤                              ◉ 7-level (bridges)     │
│      │                                                       │
│  10% ┼────────────────────────────────────────────────────  │
│      55%        60%        65%        70%        75%        │
│                    Forgetting (lower is better →)            │
│                                                              │
│  ◉ Bridges enabled   ○ Bridges disabled   ── Pareto frontier│
└─────────────────────────────────────────────────────────────┘
```

### Implementation Prompt for React Developer

```
Create a React component called DepthTradeoffScatter using D3.js.

Requirements:
1. Scatter plot with forgetting on X-axis (inverted: lower values on right) and accuracy on Y-axis
2. Point size based on number of levels: 3-level = 12px, 5-level = 18px, 7-level = 24px
3. Point color: blue (#3b82f6) for bridges enabled, gray (#9ca3af) for disabled
4. Draw dashed lines connecting same-depth configurations (3-level to 3-level, etc.)
5. If showParetoFrontier is true, draw a line connecting non-dominated points
6. On hover, show tooltip with full configuration details
7. If highlightBest is true, add a pulsing animation to the best accuracy point and best forgetting point
8. Add annotations: "Best Accuracy ↑" and "Lowest Forgetting →" with arrows
9. Include a legend showing point size meaning and color meaning
10. Add axis labels with units

Interactions:
- Click a point to "select" it and show detailed breakdown
- Hover shows tooltip
- Double-click to reset selection

Use smooth transitions for all state changes. Make it responsive.
```

---

## Component 3: BridgeFlowDiagram

### Purpose
Visualize how knowledge flows between timescale levels through the bridges. Show that this is a fully-connected graph with adaptive gating.

### Data Interface

```typescript
interface BridgeFlowDiagramProps {
  numLevels: 3 | 5 | 7;
  bridgeStats: {
    source: number;
    target: number;
    transferRate: number;  // 0-1
    avgGate: number;       // 0-1
  }[];
  animated?: boolean;
  levelNames?: string[];
}
```

### Sample Data

```javascript
const sampleData = {
  numLevels: 5,
  bridgeStats: [
    { source: 0, target: 1, transferRate: 0.95, avgGate: 0.52 },
    { source: 0, target: 2, transferRate: 0.90, avgGate: 0.48 },
    { source: 0, target: 3, transferRate: 0.85, avgGate: 0.45 },
    { source: 0, target: 4, transferRate: 0.80, avgGate: 0.42 },
    { source: 1, target: 0, transferRate: 0.92, avgGate: 0.50 },
    // ... all 20 bridges for 5-level
  ],
  levelNames: ['γ (Fast)', 'β', 'α', 'θ', 'δ (Slow)'],
};
```

### Visual Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Flow                            │
│                                                              │
│     γ (Fast)                                                 │
│        ║                                                     │
│        ║══════════════╗                                      │
│        ║              ║                                      │
│        ▼              ▼                                      │
│     β ════════════════════════════╗                         │
│        ║              ║           ║                          │
│        ║              ║           ║                          │
│        ▼              ▼           ▼                          │
│     α ════════════════════════════════════╗                 │
│        ║              ║           ║       ║                  │
│        ▼              ▼           ▼       ▼                  │
│     θ ════════════════════════════════════════════╗         │
│        ║              ║           ║       ║       ║          │
│        ▼              ▼           ▼       ▼       ▼          │
│     δ (Slow)                                                 │
│                                                              │
│  Line thickness = transfer rate   Opacity = gate value       │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Prompt for React Developer

```
Create a React component called BridgeFlowDiagram using D3.js with a Sankey-inspired layout.

Requirements:
1. Vertical layout with timescale levels as horizontal bars
2. Level bars should be labeled with brain wave names (γ, β, α, θ, δ)
3. Draw curved Bezier paths between all pairs of levels
4. Path stroke width proportional to transfer rate (1-10px range)
5. Path opacity proportional to average gate value (0.3-1.0 range)
6. Color paths by direction: blue for fast→slow, orange for slow→fast
7. If animated is true, add flowing particles along the paths:
   - Particles should be small circles (3px)
   - Speed proportional to transfer rate
   - Spawn rate proportional to gate value
8. On hover over a level, highlight all its incoming and outgoing connections
9. On hover over a path, show tooltip with exact transfer rate and gate value
10. Add a legend explaining line thickness and opacity

The visualization should clearly show that:
- All levels are connected to all other levels (full connectivity)
- Some connections are stronger than others (adaptive gating)
- Information flows in both directions (bidirectional)

Make it responsive and add a "Pause Animation" button.
```

---

## Component 4: ArchitectureComparisonCards

### Purpose
Provide a quick visual comparison of the three architecture depths (3, 5, 7 levels) with key metrics.

### Data Interface

```typescript
interface ArchitectureCardProps {
  levels: number;
  levelNames: string[];
  frequencies: number[];
  bridgeCount: number;
  accuracy: { bridges: number; noBridges: number };
  forgetting: { bridges: number; noBridges: number };
  recommended?: boolean;
}
```

### Visual Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Architecture Comparison                              │
│                                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │    3-Level       │  │    5-Level ⭐    │  │    7-Level       │          │
│  │   γ → β → α      │  │ γ → β → α → θ → δ│  │ γ → β → α → θ → │          │
│  │                  │  │                  │  │   δ → ε → ζ     │          │
│  │  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────────┐  │          │
│  │  │████████████│  │  │  │████████████│  │  │  │████████████│  │          │
│  │  │  ████████  │  │  │  │ █████████ │  │  │  │ ██████████ │  │          │
│  │  │    ████    │  │  │  │  ███████  │  │  │  │  ████████  │  │          │
│  │  └────────────┘  │  │  │   █████   │  │  │  │   ██████   │  │          │
│  │                  │  │  │    ███    │  │  │  │    ████    │  │          │
│  │  Frequencies:    │  │  └────────────┘  │  │  │     ██     │  │          │
│  │  1, 10, 100      │  │                  │  │  │      █     │  │          │
│  │                  │  │  Frequencies:    │  │  └────────────┘  │          │
│  │  Bridges: 6      │  │  1, 5, 25, 125,  │  │                  │          │
│  │                  │  │  625             │  │  Frequencies:    │          │
│  │  Accuracy:       │  │                  │  │  1, 3, 9, 27,    │          │
│  │  17.8% / 15.5%   │  │  Bridges: 20     │  │  81, 243, 729    │          │
│  │                  │  │                  │  │                  │          │
│  │  Forgetting:     │  │  Accuracy:       │  │  Bridges: 42     │          │
│  │  70.0% / 64.7%   │  │  16.3% / 18.5%   │  │                  │          │
│  │                  │  │                  │  │  Accuracy:       │          │
│  │                  │  │  Forgetting:     │  │  13.6% / 12.9%   │          │
│  │                  │  │  69.9% / 73.8%   │  │                  │          │
│  │                  │  │                  │  │  Forgetting:     │          │
│  │                  │  │                  │  │  56.9% / 56.5%   │          │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘          │
│                                                                              │
│  Format: Without Bridges / With Bridges                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Prompt for React Developer

```
Create a React component called ArchitectureComparisonCards using Tailwind CSS.

Requirements:
1. Three cards in a responsive row (stack on mobile)
2. Each card contains:
   a. Header with level count and brain wave progression (γ → β → α...)
   b. Visual representation: stacked horizontal bars of decreasing width
      - Each bar represents a timescale level
      - Width proportional to update frequency (fast = wide, slow = narrow)
      - Color gradient from blue (fast) to purple (slow)
   c. Frequency list showing update intervals
   d. Bridge count with a small network icon
   e. Accuracy comparison: "Without / With Bridges"
   f. Forgetting comparison: "Without / With Bridges"
   g. Color-code improvements: green if bridges help, red if they hurt
3. If recommended is true, add a star badge and subtle glow effect
4. On hover, expand the card slightly and show additional details
5. Add a toggle to switch between "Accuracy Focus" and "Forgetting Focus" views
6. Include tooltips explaining each metric

Styling:
- Use a clean, modern card design with subtle shadows
- Consistent spacing and typography
- Accessible color contrast
- Smooth hover transitions

The 5-level card should be marked as "Recommended" with a star icon.
```

---

## Component 5: InteractiveCodeExplorer

### Purpose
Let visitors explore the codebase with syntax highlighting and annotations.

### Data Interface

```typescript
interface CodeExplorerProps {
  files: {
    path: string;
    content: string;
    language: 'python' | 'typescript';
    annotations?: {
      line: number;
      text: string;
      type: 'info' | 'important' | 'novel';
    }[];
    highlighted?: boolean;
  }[];
  initialFile?: string;
}
```

### Visual Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Code Explorer                                      │
│                                                                              │
│  ┌──────────────────┐  ┌────────────────────────────────────────────────┐  │
│  │ 📁 src/          │  │ knowledge_bridges.py                    [Copy] │  │
│  │   📁 optimizers/ │  ├────────────────────────────────────────────────┤  │
│  │     📄 deep_mom..│  │  1 │ """Knowledge Bridges: Bidirectional...   │  │
│  │     📄 nested_o..│  │  2 │                                          │  │
│  │     📄 multi_sc..│  │  3 │ This is our NOVEL contribution...        │  │
│  │   📁 bridges/    │  │    │ ┌─────────────────────────────────────┐  │  │
│  │     ⭐ knowledge.│  │    │ │ 💡 NOVEL: The paper only has        │  │  │
│  │   📁 memory/     │  │    │ │ unidirectional flow. We add         │  │  │
│  │     📄 memory_b..│  │    │ │ explicit bidirectional bridges.     │  │  │
│  │     📄 continuum.│  │    │ └─────────────────────────────────────┘  │  │
│  │ 📁 tests/        │  │ 72 │ class KnowledgeBridge(nn.Module):       │  │
│  │ 📁 benchmarks/   │  │ 73 │     """Attention-gated knowledge...     │  │
│  │                  │  │ 74 │                                          │  │
│  │                  │  │ 75 │     def forward(self, source, target):  │  │
│  │                  │  │    │ ┌─────────────────────────────────────┐  │  │
│  │                  │  │    │ │ 🔑 KEY: Similarity-based gating     │  │  │
│  │                  │  │    │ │ provides adaptive transfer without  │  │  │
│  │                  │  │    │ │ requiring training.                 │  │  │
│  │                  │  │    │ └─────────────────────────────────────┘  │  │
│  │                  │  │ 76 │         similarity = cosine_sim(...)    │  │
│  │                  │  │ 77 │         gate = (similarity + 1) / 2     │  │
│  └──────────────────┘  └────────────────────────────────────────────────┘  │
│                                                                              │
│  ⭐ = Key file   💡 = Novel contribution   🔑 = Important concept           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Prompt for React Developer

```
Create a React component called InteractiveCodeExplorer with syntax highlighting.

Requirements:
1. Left panel: Collapsible file tree
   - Show folder structure with icons (📁 for folders, 📄 for files)
   - Star icon (⭐) for key files
   - Click to select file
   - Highlight currently selected file
2. Right panel: Code viewer
   - Syntax highlighting using Prism.js or highlight.js
   - Line numbers
   - Copy button in header
   - Smooth scroll to annotations
3. Annotations:
   - Inline callout boxes that appear between lines
   - Three types with different colors:
     - info (blue): General explanations
     - important (yellow): Key concepts
     - novel (green): Our contributions
   - Collapsible (click to expand/collapse)
4. Navigation:
   - "Next Annotation" / "Previous Annotation" buttons
   - Jump to specific line number
   - Search within file
5. Responsive:
   - On mobile, file tree becomes a dropdown
   - Code viewer takes full width

Key files to highlight:
- src/bridges/knowledge_bridges.py (NOVEL)
- src/optimizers/deep_momentum.py
- src/optimizers/multi_scale.py
- src/memory/continuum.py

Add annotations explaining:
- The similarity-based gating mechanism
- How knowledge injection works
- The multi-timescale update logic
```

---

## Component 6: AnimatedLearningTimeline

### Purpose
Animate the training process to show how accuracy evolves and forgetting happens in real-time.

### Data Interface

```typescript
interface LearningTimelineProps {
  data: {
    step: number;
    taskAccuracies: number[];  // Accuracy for each task at this step
    currentTask: number;       // Which task is being trained
  }[];
  taskNames: string[];
  speed?: number;  // Animation speed multiplier
}
```

### Visual Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Learning Timeline                                     │
│                                                                              │
│  100% ┤     ╭──╮                                                            │
│       │    ╱    ╲         ╭──╮                                              │
│   80% ┤   ╱      ╲       ╱    ╲                                             │
│       │  ╱        ╲     ╱      ╲        ╭──╮                                │
│   60% ┤ ╱          ╲   ╱        ╲      ╱    ╲                               │
│       │╱            ╲ ╱          ╲    ╱      ╲                              │
│   40% ┤              ╳            ╲  ╱        ╲                             │
│       │             ╱ ╲            ╲╱          ╲                            │
│   20% ┤────────────╱   ╲───────────────────────╲────────────               │
│       │                                                                      │
│    0% ┼──────────┬──────────┬──────────┬──────────┬──────────               │
│       0         100        200        300        400        500             │
│                           Training Steps                                     │
│                                                                              │
│  ── Task 1   ── Task 2   ── Task 3   ── Task 4   ── Task 5                 │
│                                                                              │
│  [▶ Play]  [⏸ Pause]  [⏮ Reset]  Speed: [1x ▼]  Step: 247/500             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Prompt for React Developer

```
Create a React component called AnimatedLearningTimeline using D3.js.

Requirements:
1. Line chart with training steps on X-axis and accuracy on Y-axis
2. One line per task, each with a distinct color
3. Lines should animate from left to right as "training progresses"
4. Vertical markers at task boundaries (when training switches to new task)
5. Current training task should have a thicker, more prominent line
6. When a new task starts training:
   - Its line should rise quickly (learning)
   - Previous task lines should fall (forgetting)
7. Controls:
   - Play/Pause button
   - Reset button
   - Speed slider (0.5x, 1x, 2x, 4x)
   - Step counter showing current/total
8. On hover, show tooltip with all task accuracies at that step
9. Add a "forgetting indicator" that shows when accuracy drops significantly
10. Include a legend with task colors

Animation details:
- Default speed: 50ms per step
- Smooth line interpolation (use D3 curve)
- Fade in new task lines when they start
- Pulse effect when forgetting is detected (>10% drop)

Make it responsive and add keyboard controls (space for play/pause, arrow keys for step).
```

---

## Component 7: ResultsSummaryDashboard

### Purpose
A comprehensive dashboard that ties all the visualizations together with key takeaways.

### Visual Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Collaborative Nested Learning                             │
│           Bidirectional Knowledge Bridges for Continual Learning            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         Key Findings                                     ││
│  │                                                                          ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ ││
│  │  │    +2.2%     │  │    -13%      │  │     109      │  │      6       │ ││
│  │  │  Accuracy    │  │  Forgetting  │  │    Tests     │  │   Bridges    │ ││
│  │  │  (5-level)   │  │  (7-level)   │  │   Passing    │  │   (3-level)  │ ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌────────────────────────────────┐  ┌────────────────────────────────────┐ │
│  │      Forgetting Heatmap        │  │     Depth vs. Performance          │ │
│  │                                │  │                                    │ │
│  │      [Heatmap Component]       │  │    [Scatter Plot Component]        │ │
│  │                                │  │                                    │ │
│  └────────────────────────────────┘  └────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────┐  ┌────────────────────────────────────┐ │
│  │      Bridge Flow Diagram       │  │     Architecture Cards             │ │
│  │                                │  │                                    │ │
│  │    [Flow Diagram Component]    │  │    [Cards Component]               │ │
│  │                                │  │                                    │ │
│  └────────────────────────────────┘  └────────────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      Animated Learning Timeline                          ││
│  │                                                                          ││
│  │                    [Timeline Component]                                  ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  📖 Read the Blog Post    💻 View the Code    📄 Read the Paper         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Prompt for React Developer

```
Create a React dashboard component called ResultsSummaryDashboard that composes all the visualization components.

Requirements:
1. Hero section with title and subtitle
2. Key metrics row with 4 stat cards:
   - "+2.2% Accuracy" (green, with up arrow)
   - "-13% Forgetting" (green, with down arrow)
   - "109 Tests Passing" (blue, with checkmark)
   - "6 Bridges" (purple, with network icon)
3. 2x2 grid of visualizations:
   - Top left: ForgettingHeatmap
   - Top right: DepthTradeoffScatter
   - Bottom left: BridgeFlowDiagram
   - Bottom right: ArchitectureComparisonCards
4. Full-width AnimatedLearningTimeline below the grid
