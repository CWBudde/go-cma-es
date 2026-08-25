"use strict";

Demo.start(function (info) {
  const stage = Demo.$("stage");
  const select = Demo.$("landscape");
  const landscapeSinks = {};
  const runSinks = {};
  let landscape;
  let image;
  let run;

  for (const spec of info.landscapes) {
    const option = document.createElement("option");
    option.value = spec.key;
    option.textContent = spec.name;
    select.appendChild(option);
  }

  const transport = Demo.Transport(draw);

  function options() {
    return {
      landscape: select.value,
      seed: Number(Demo.$("seed").value),
      lambda: Number(Demo.$("lambda").value),
      sigma: Number(Demo.$("sigma").value),
      iterations: Number(Demo.$("iterations").value),
      active: Demo.$("active").checked,
      mode: Demo.$("mode").value,
      out: runSinks,
    };
  }

  function draw(frame) {
    if (!run || !landscape) return;
    Render.run(stage, landscape, image, run, frame);
    Demo.$("tGeneration").textContent = String(frame + 1);
    Demo.$("tBest").textContent = Number(run.bestCost[frame]).toExponential(4);
    Demo.$("tSigma").textContent = Number(run.sigma[frame]).toExponential(3);
    Demo.$("tCondition").textContent = Number(run.condition[frame]).toExponential(3);
  }

  function loadLandscape() {
    landscape = Demo.call("landscape", {
      landscape: select.value,
      width: 180,
      height: 180,
      out: landscapeSinks,
    });
    if (!landscape) return false;
    Demo.cacheSinks(landscapeSinks, landscape, ["values"]);
    image = Render.backdrop(landscape);
    Demo.$("landscapeNote").textContent = landscape.note;
    return true;
  }

  function execute(autoplay) {
    transport.stop();
    Demo.setStatus("Computing the run…", "loading");
    if (!loadLandscape()) return;
    run = Demo.call("run", options());
    if (!run) return;
    Demo.cacheSinks(runSinks, run, ["population", "mean", "ellipse", "bestTrail", "bestCost", "iterationBest", "sigma", "condition"]);
    Demo.$("tEvals").textContent = String(run.evaluations);
    Demo.$("tTermination").textContent = run.termination;
    transport.reset(run.frames, autoplay);
    Demo.setStatus(
      autoplay ? "Playing the run — scrub or pause any time." : "Ready — drag the timeline or press play.",
      "ready",
    );
  }

  select.addEventListener("change", () => {
    const spec = info.landscapes.find((entry) => entry.key === select.value);
    Demo.$("sigma").value = spec.sigma;
    execute();
  });
  Demo.$("run").addEventListener("click", () => execute(true));
  execute();
});
