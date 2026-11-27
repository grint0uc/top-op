---
name: Bug report
about: Something crashes, gives a wrong result, or the GUI misbehaves
title: ""
labels: bug
assignees: ""
---

**What happened**
A short description, and what you expected instead.

**How to reproduce**
The smallest setup that shows it. Best: a case file that `topop run` reproduces.

- From the GUI: Results tab -> Export -> `project.json`. From the CLI: the case file, or the `run.json`
  that `topop run` wrote to its output directory.
- Attach the mesh (STL / STEP / 3MF) the case refers to. If it is large or confidential, a simplified part
  that still shows the problem is fine. Zip files if GitHub refuses the extension.
- If it is a selection problem, the output of `topop describe your-part.stl` helps.

```
command you ran, e.g. uv run topop run case.json --max-iter 5
```

**Output**
The error message or the last lines of the CLI output (or the browser console / server log for GUI issues).

```
paste here
```

**Environment**
- top-op version (`uv run python -c "import topop; print(topop.__version__)"`) or commit:
- Install method (clone + uv sync, uv tool, pipx, pip):
- OS and CPU / RAM:
- Python version:
- Browser (GUI issues):
- Optional extras installed (step, accel):
