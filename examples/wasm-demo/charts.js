"use strict";

Demo.start(function (info) {
  const select = Demo.$("landscape");
  const sinks = {};
  let result;

  for (const spec of info.landscapes) {
    const option = document.createElement("option");
    option.value = spec.key;
    option.textContent = spec.name;
    select.appendChild(option);
  }

  const transport = Demo.Transport(draw);

  function draw(frame) {
    if (!result) return;
    Render.chart(Demo.$("costChart"), [
      { values: result.bestCost, color: Render.palette.best },
      { values: result.iterationBest, color: Render.palette.mean },
    ], frame, true);
    Render.chart(Demo.$("stateChart"), [
      { values: result.sigma, color: Render.palette.ellipse },
      { values: result.condition, color: Render.palette.baseline },
    ], frame, true);
    Demo.$("readout").textContent = `best ${Number(result.bestCost[frame]).toExponential(3)} · σ ${Number(result.sigma[frame]).toExponential(3)} · cond ${Number(result.condition[frame]).toExponential(3)}`;
  }

  function execute() {
    Demo.setStatus("Computing telemetry…", "loading");
    result = Demo.call("run", {
      landscape: select.value,
      seed: Number(Demo.$("seed").value),
      lambda: Number(Demo.$("lambda").value),
      sigma: Number(Demo.$("sigma").value),
      iterations: Number(Demo.$("iterations").value),
      active: Demo.$("active").checked,
      mode: Demo.$("mode").value,
      out: sinks,
    });
    if (!result) return;
    Demo.cacheSinks(sinks, result, ["population", "mean", "ellipse", "bestTrail", "bestCost", "iterationBest", "sigma", "condition"]);
    transport.reset(result.frames);
    Demo.setStatus("Ready — both charts share the generation cursor.", "ready");
  }

  select.addEventListener("change", () => {
    const spec = info.landscapes.find((entry) => entry.key === select.value);
    Demo.$("sigma").value = spec.sigma;
  });
  Demo.$("run").addEventListener("click", execute);
  execute();
});
