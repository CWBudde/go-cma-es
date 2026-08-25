"use strict";

Demo.start(function () {
  const landscapeSinks = {};
  const restartSinks = {};
  let landscape;
  let image;

  function execute() {
    Demo.setStatus("Running the population-doubling schedule…", "loading");
    landscape = Demo.call("landscape", { landscape: "rastrigin", width: 180, height: 180, out: landscapeSinks });
    if (!landscape) return;
    Demo.cacheSinks(landscapeSinks, landscape, ["values"]);
    image = Render.backdrop(landscape);

    const result = Demo.call("restart", {
      seed: Number(Demo.$("seed").value),
      lambda: Number(Demo.$("lambda").value),
      iterations: Number(Demo.$("iterations").value),
      restarts: Number(Demo.$("restarts").value),
      out: restartSinks,
    });
    if (!result) return;
    Demo.cacheSinks(restartSinks, result, ["markers"]);
    Render.restarts(Demo.$("restartStage"), landscape, image, result.markers);
    Demo.$("totalEvals").textContent = String(result.totalEvaluations);
    Demo.$("globalBest").textContent = Number(result.best).toExponential(4);

    const body = Demo.$("restartRows");
    body.replaceChildren();
    for (const record of result.records) {
      const row = document.createElement("tr");
      const values = [record.restart, record.lambda, record.iterations, record.evaluations, `(${record.basin[0]}, ${record.basin[1]})`, Number(record.best).toExponential(3), record.termination];
      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.appendChild(cell);
      }
      body.appendChild(row);
    }
    Demo.setStatus("Ready — numbered points are restart winners; violet squares mark their nearest Rastrigin basin.", "ready");
  }

  Demo.$("run").addEventListener("click", execute);
  execute();
});
