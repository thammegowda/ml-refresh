import { select, scaleLinear, axisBottom, axisLeft, extent, line, area, zoom, zoomIdentity, bisector } from 'd3';

export const curveColors = ['#087f72', '#b04e75', '#a76613', '#4169a1', '#6b5792', '#8b4b32', '#277981', '#56672d'];
export const curveDashes = ['', '6 4', '2 3', '9 3 2 3', '12 4', '4 2', '1 3', '8 2 2 2'];

export function fitPlotDomains(plot) {
  const points = plot.series.flatMap(series => series.points);
  const domain = extent(points, entry => entry[0]);
  if (domain[0] === domain[1]) { domain[0] -= 1; domain[1] += 1; }
  const finite = points.filter(entry => Number.isFinite(entry[1]));
  let [minimum, maximum] = extent(finite, entry => entry[1]);
  minimum = Math.min(0, minimum ?? -1);
  maximum = Math.max(0, maximum ?? 1);
  const padding = (maximum - minimum || 1) * 0.1;
  return { x: domain, y: scaleLinear().domain([minimum - padding, maximum + padding]).nice().domain() };
}

export function renderPlot(container, { plot, domains, id, view, hidden, onViewChange, changeZoom, point, onPointChange }) {
  if (!container?.clientWidth) return;
  const width = container.clientWidth;
  const height = width * 9 / 16;
  const margin = { top: 14, right: 16, bottom: 32, left: 54 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const ticksX = Math.max(4, Math.floor(plotWidth / 45));
  const ticksY = Math.max(4, Math.floor(plotHeight / 35));
  const transform = zoomIdentity.translate(margin.left * (1 - view.k) + view.x * plotWidth, margin.top * (1 - view.k) + view.y * plotHeight).scale(view.k);
  const scaleX = transform.rescaleX(scaleLinear().domain(domains.x).range([margin.left, width - margin.right]));
  const scaleY = transform.rescaleY(scaleLinear().domain(domains.y).range([height - margin.bottom, margin.top]));
  const svg = select(container).selectAll('svg').data([null]).join('svg').attr('viewBox', `0 0 ${width} ${height}`).attr('role', point ? 'group' : 'img').attr('tabindex', 0).attr('aria-label', `${plot.title} plot`).attr('data-zoom', view.k);
  const behavior = zoom().scaleExtent([0.25, 32]).extent([[margin.left, margin.top], [width - margin.right, height - margin.bottom]])
    .filter(event => !event.target.closest('.py-point-control') && svg.property('__pointPointer') == null && (!event.ctrlKey || event.type === 'wheel') && !event.button)
    .on('zoom', event => {
      const next = event.transform;
      onViewChange({ k: next.k, x: (next.x - margin.left * (1 - next.k)) / plotWidth, y: (next.y - margin.top * (1 - next.k)) / plotHeight });
    });
  svg.property('__zoom', transform).call(behavior).on('keydown.python-plot', event => {
    if (['+', '=', '-', '0'].includes(event.key)) {
      event.preventDefault();
      changeZoom(event.key === '0' ? null : event.key === '-' ? 1 / 1.5 : 1.5);
    } else if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) {
      event.preventDefault();
      onViewChange({ ...view, x: view.x + (event.key === 'ArrowLeft' ? 0.1 : event.key === 'ArrowRight' ? -0.1 : 0), y: view.y + (event.key === 'ArrowUp' ? 0.1 : event.key === 'ArrowDown' ? -0.1 : 0) });
    }
  });
  svg.selectAll('*').filter(function () { return !this.closest('.py-point-control'); }).remove();
  svg.append('title').text(plot.title);
  svg.append('defs').append('clipPath').attr('id', id).append('rect').attr('x', margin.left).attr('y', margin.top).attr('width', plotWidth).attr('height', plotHeight);
  svg.append('g').attr('class', 'py-plot-grid').attr('transform', `translate(0,${height - margin.bottom})`).call(axisBottom(scaleX).ticks(ticksX).tickSize(-plotHeight).tickFormat(''));
  svg.append('g').attr('class', 'py-plot-grid').attr('transform', `translate(${margin.left},0)`).call(axisLeft(scaleY).ticks(ticksY).tickSize(-plotWidth).tickFormat(''));
  svg.append('g').attr('class', 'py-plot-minor').attr('transform', `translate(0,${height - margin.bottom})`).call(axisBottom(scaleX).ticks(ticksX * 2).tickSize(3).tickFormat(''));
  svg.append('g').attr('class', 'py-plot-minor').attr('transform', `translate(${margin.left},0)`).call(axisLeft(scaleY).ticks(ticksY * 2).tickSize(3).tickFormat(''));
  svg.append('g').attr('class', 'py-plot-axis').attr('transform', `translate(0,${height - margin.bottom})`).call(axisBottom(scaleX).ticks(ticksX));
  svg.append('g').attr('class', 'py-plot-axis').attr('transform', `translate(${margin.left},0)`).call(axisLeft(scaleY).ticks(ticksY));
  const graph = svg.append('g').attr('clip-path', `url(#${id})`);
  const drawArea = area().defined(entry => Number.isFinite(entry[1])).x(entry => scaleX(entry[0])).y0(scaleY(0)).y1(entry => scaleY(entry[1]));
  plot.series.forEach((series, index) => {
    const region = series.area;
    if (!region || region.value === null || region.lower === region.upper || hidden.has(series.name)) return;
    const left = Math.max(margin.left, scaleX(Math.min(region.lower, region.upper)));
    const right = Math.min(width - margin.right, scaleX(Math.max(region.lower, region.upper)));
    const zero = Math.max(margin.top, Math.min(height - margin.bottom, scaleY(0)));
    const direction = region.upper >= region.lower ? 1 : -1;
    for (const sign of [1, -1]) {
      const above = sign * direction > 0;
      const clip = `${id}-area-${index}-${sign}`;
      svg.select('defs').append('clipPath').attr('id', clip).append('rect')
        .attr('x', left).attr('y', above ? margin.top : zero).attr('width', Math.max(0, right - left))
        .attr('height', above ? zero - margin.top : height - margin.bottom - zero);
      graph.append('path').attr('class', `py-area py-area-${sign > 0 ? 'positive' : 'negative'}`)
        .attr('clip-path', `url(#${clip})`).attr('fill', curveColors[sign > 0 ? 0 : 1]).attr('fill-opacity', 0.24).attr('d', drawArea(series.points));
    }
    for (const bound of [region.lower, region.upper]) {
      graph.append('line').attr('class', 'py-area-bound').attr('x1', scaleX(bound)).attr('x2', scaleX(bound))
        .attr('y1', margin.top).attr('y2', height - margin.bottom).attr('stroke', '#64716d').attr('stroke-dasharray', '3 3');
    }
  });
  graph.append('line').attr('class', 'py-zero-axis').attr('x1', margin.left).attr('x2', width - margin.right).attr('y1', scaleY(0)).attr('y2', scaleY(0));
  const drawLine = line().defined(entry => Number.isFinite(entry[1])).x(entry => scaleX(entry[0])).y(entry => scaleY(entry[1]));
  plot.series.forEach((series, index) => {
    if (hidden.has(series.name)) return;
    graph.append('path').datum(series.points).attr('class', 'py-curve').attr('data-series', series.name).attr('fill', 'none').attr('stroke', curveColors[index]).attr('stroke-dasharray', curveDashes[index]).attr('stroke-width', 2).attr('d', drawLine);
  });
  const handle = svg.selectAll('.py-point-control').data(point ? [point] : []).join(enter => {
    const group = enter.append('g').attr('class', 'py-point-control');
    group.append('title');
    group.append('circle').attr('class', 'py-point-hit').attr('r', 8);
    group.append('circle').attr('class', 'py-point-dot').attr('r', 6);
    return group;
  });
  if (point) {
    const samples = plot.series.find(series => !hidden.has(series.name) && series.name !== 'Tangent')?.points ?? [];
    const index = bisector(entry => entry[0]).left(samples, point.value);
    const before = samples[Math.max(0, index - 1)];
    const after = samples[Math.min(samples.length - 1, index)];
    const ordinate = before && after && Number.isFinite(before[1]) && Number.isFinite(after[1])
      ? before[0] === after[0] ? before[1] : before[1] + (after[1] - before[1]) * (point.value - before[0]) / (after[0] - before[0]) : null;
    const setPoint = value => {
      if (!point.disabled) onPointChange(Math.max(point.min, Math.min(point.max, Math.round(value * 100) / 100)));
    };
    const movePoint = event => {
      if (svg.property('__pointPointer') !== event.pointerId) return;
      const matrix = svg.node().getScreenCTM();
      if (!matrix) return;
      const position = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
      setPoint(scaleX.invert(position.x));
    };
    const endPoint = event => {
      if (svg.property('__pointPointer') !== event.pointerId) return;
      if (event.type === 'pointerup') movePoint(event);
      svg.property('__pointPointer', null);
      if (svg.node().hasPointerCapture(event.pointerId)) svg.node().releasePointerCapture(event.pointerId);
    };
    handle.raise().attr('role', 'slider').attr('tabindex', point.disabled ? -1 : 0)
      .attr('aria-label', 'Plot evaluation point').attr('aria-valuemin', point.min).attr('aria-valuemax', point.max)
      .attr('aria-valuenow', point.value).attr('aria-valuetext', `x = ${point.value}${ordinate === null ? ', undefined' : ''}`).attr('aria-disabled', point.disabled)
      .attr('transform', `translate(${Math.max(margin.left + 8, Math.min(width - margin.right - 8, scaleX(point.value)))} ${Math.max(margin.top + 8, Math.min(height - margin.bottom - 8, scaleY(ordinate ?? 0)))})`)
      .classed('py-point-undefined', ordinate === null)
      .on('pointerdown.python-point', event => {
        if (event.button !== 0 || point.disabled || svg.property('__pointPointer') != null) return;
        event.preventDefault();
        event.stopPropagation();
        handle.node().focus();
        svg.property('__pointPointer', event.pointerId);
        svg.node().setPointerCapture(event.pointerId);
      })
      .on('keydown.python-point', event => {
        if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        event.stopPropagation();
        const step = event.shiftKey ? 0.1 : 0.01;
        setPoint(event.key === 'Home' ? point.min : event.key === 'End' ? point.max : point.value + (['ArrowLeft', 'ArrowDown'].includes(event.key) ? -step : step));
      });
    handle.select('title').text('Drag to change the evaluation point; arrow keys adjust by 0.01');
    svg.on('pointermove.python-point', movePoint).on('pointerup.python-point pointercancel.python-point lostpointercapture.python-point', endPoint);
  } else svg.on('.python-point', null);
  if (!plot.series.some(series => series.points.some(entry => Number.isFinite(entry[1])))) svg.append('text').attr('class', 'py-plot-empty').attr('x', width / 2).attr('y', height / 2).attr('text-anchor', 'middle').text('No finite values');
}

export function detachPlot(container) {
  if (container) select(container).select('svg').on('.zoom', null).on('.python-plot', null).on('.python-point', null);
}