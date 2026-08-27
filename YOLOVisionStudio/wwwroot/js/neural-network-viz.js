// D3.js Neural Network Visualizer
window.initNetworkGraph = function(containerId, architectureData) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Fallback YOLO11n Data
    if (!architectureData) {
        architectureData = {
            layers: [
                {id: 0, name: "Conv_0", type: "Conv", section: "backbone", parameters: 896},
                {id: 1, name: "Conv_1", type: "Conv", section: "backbone", parameters: 18000},
                {id: 2, name: "C2F_2", type: "C2F", section: "backbone", parameters: 50000},
                {id: 3, name: "SPPF_3", type: "SPPF", section: "backbone", parameters: 100000},
                {id: 4, name: "UPSAMPLE_4", type: "Upsample", section: "neck", parameters: 0},
                {id: 5, name: "CONCAT_5", type: "Concat", section: "neck", parameters: 0},
                {id: 6, name: "Detect", type: "Detect", section: "head", parameters: 200000}
            ],
            edges: [
                {from: 0, to: 1}, {from: 1, to: 2}, {from: 2, to: 3}, 
                {from: 3, to: 4}, {from: 4, to: 5}, {from: 2, to: 5}, 
                {from: 5, to: 6}
            ]
        };
    }

    const width = container.clientWidth;
    const height = container.clientHeight;

    container.innerHTML = '';
    const svg = d3.select("#" + containerId).append("svg")
        .attr("width", width)
        .attr("height", height);

    const zoom = d3.zoom().on("zoom", (e) => g.attr("transform", e.transform));
    svg.call(zoom);

    const g = svg.append("g");

    // Colors
    const colorScale = d3.scaleOrdinal()
        .domain(["Conv", "C2F", "SPPF", "Upsample", "Concat", "Detect"])
        .range(["#7c3aed", "#06b6d4", "#10b981", "#3b82f6", "#94a3b8", "#f59e0b"]);

    // Layout calculation (simple vertical tree layout mock)
    architectureData.layers.forEach((d, i) => {
        d.x = width / 2;
        d.y = 50 + i * 80;
    });

    // Links
    g.selectAll(".link")
        .data(architectureData.edges)
        .enter().append("line")
        .attr("class", "link")
        .attr("x1", d => architectureData.layers.find(l=>l.id===d.from).x)
        .attr("y1", d => architectureData.layers.find(l=>l.id===d.from).y)
        .attr("x2", d => architectureData.layers.find(l=>l.id===d.to).x)
        .attr("y2", d => architectureData.layers.find(l=>l.id===d.to).y);

    // Nodes
    const node = g.selectAll(".node")
        .data(architectureData.layers)
        .enter().append("g")
        .attr("class", "node")
        .attr("transform", d => `translate(${d.x},${d.y})`)
        .on("click", (e, d) => {
            // Expand details (Mock)
            if (window.showToast) window.showToast("Layer Info", `Selected ${d.name} (${d.type})`, "info");
        });

    node.append("circle")
        .attr("r", d => Math.max(15, Math.min(30, Math.sqrt(d.parameters)/10)))
        .style("fill", d => colorScale(d.type));

    node.append("text")
        .attr("dy", 4)
        .attr("dx", 35)
        .text(d => d.name);
};
