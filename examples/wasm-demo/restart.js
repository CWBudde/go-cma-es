"use strict";

Demo.start(function (info) {
  const select = Demo.$("landscape");
  const landscapeSinks = {};
  const restartSinks = {};
  let landscape;
  let image;

  for (const spec of info.landscapes) {
    const option = document.createElement("option");
    option.value = spec.key;
    option.textContent = spec.name;
    select.appendChild(option);
  }
  select.value = info.landscapes.some((spec) => spec.key === "schaffer") ? "schaffer" : select.value;

  function coordinate(value) {
    return Number.isInteger(value) ? String(value) : Number(value).toFixed(2);
  }

  function execute() {
    Demo.setStatus("Running the population-doubling schedule…", "loading");
    landscape = Demo.call("landscape", { landscape: select.value, width: 180, height: 180, out: landscapeSinks });
    if (!landscape) return;
    Demo.cacheSinks(landscapeSinks, landscape, ["values"]);
    image = Render.backdrop(landscape);
    Demo.$("landscapeNote").textContent = landscape.note;

    const result = Demo.call("restart", {
      landscape: select.value,
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

    const label = result.basinLabel || "run winner";
    Demo.$("stageCaption").textContent = `${result.landscapeName} · rank shaded`;
    Demo.$("basinHeading").textContent = result.basinLabel ? "Basin" : "Winner";
    Demo.$("basinLegend").textContent = label;

    const body = Demo.$("restartRows");
    body.replaceChildren();
    for (const record of result.records) {
      const row = document.createElement("tr");
      const values = [record.restart, record.lambda, record.iterations, record.evaluations, `(${coordinate(record.basin[0])}, ${coordinate(record.basin[1])})`, Number(record.best).toExponential(3), record.termination];
      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.appendChild(cell);
      }
      body.appendChild(row);
    }
    Demo.setStatus(`Ready — numbered points are restart winners; violet squares mark the ${label}.`, "ready");
  }

  Demo.$("run").addEventListener("click", execute);
  select.addEventListener("change", execute);
  execute();
});
