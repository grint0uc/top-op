// Panel names follow docs/PLAN.md section 4; each becomes a ui/*Panel component in A3.
const PANELS = ['Import', 'Domain', 'Loads', 'Supports', 'Reference models', 'Run', 'Results'];

export function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <h1 className="brand">top-op</h1>
        {PANELS.map((name) => (
          <section className="panel" key={name}>
            <h2>{name}</h2>
          </section>
        ))}
      </aside>
      <main className="main">
        <canvas id="viewport" />
      </main>
    </div>
  );
}
