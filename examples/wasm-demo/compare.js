"use strict";

Demo.start(function () {
  const landscapeSinks = {};
  const comparisonSinks = {};
  let landscape;
  let image;
  let comparison;

  const transport = Demo.Transport(draw);

  function draw(frame) {
    if (!comparison) return;
    Render.run(Demo.$("cmaStage"), landscape, image, {
      population: comparison.cmaPopulation,
      mean: comparison.cmaMean,
      ellipse: comparison.cmaEllipse,
      bestTrail: comparison.cmaBestTrail,
      lambda: comparison.cmaLambda,
    }, Math.min(frame, comparison.cmaFrames - 1));
    Render.run(Demo.$("isoStage"), landscape, image, {
      population: comparison.isoPopulation,
      mean: comparison.isoMean,
      bestTrail: comparison.isoBestTrail,
      lambda: comparison.isoLambda,
    }, Math.min(frame, comparison.isoFrames - 1), {
      samples: "rgba(227,239,242,.7)",
      mean: Render.palette.baseline,
      trail: Render.palette.baseline,
    });
    Demo.$("cmaLive").textContent = Number(comparison.cmaBestCost[Math.min(frame, comparison.cmaBestCost.length - 1)]).toExponential(3);
    Demo.$("isoLive").textContent = Number(comparison.isoBestCost[Math.min(frame, comparison.isoBestCost.length - 1)]).toExponential(3);
  }

  function execute(autoplay) {
    Demo.setStatus("Computing paired searches…", "loading");
    landscape = Demo.call("landscape", { landscape: "ellipsoid", width: 180, height: 180, out: landscapeSinks });
    if (!landscape) return;
    Demo.cacheSinks(landscapeSinks, landscape, ["values"]);
    image = Render.backdrop(landscape);

    comparison = Demo.call("compare", {
      seed: Number(Demo.$("seed").value),
      lambda: Number(Demo.$("lambda").value),
      sigma: Number(Demo.$("sigma").value),
      iterations: Number(Demo.$("iterations").value),
      out: comparisonSinks,
    });
    if (!comparison) return;
    Demo.cacheSinks(comparisonSinks, comparison, ["cmaPopulation", "cmaMean", "cmaEllipse", "cmaBestTrail", "cmaBestCost", "cmaIterationBest", "cmaSigma", "cmaCondition", "isoPopulation", "isoMean", "isoBestTrail", "isoBestCost"]);
    Demo.$("cmaFinal").textContent = Number(comparison.cmaBest).toExponential(4);
    Demo.$("isoFinal").textContent = Number(comparison.isoBest).toExponential(4);
    Demo.$("budget").textContent = `${comparison.budget} evaluations each`;
    transport.reset(Math.max(comparison.cmaFrames, comparison.isoFrames), autoplay);
    Demo.setStatus("Ready — identical normal draws, identical evaluation budget.", "ready");
  }

  Demo.$("run").addEventListener("click", () => execute(true));
  execute();
});
