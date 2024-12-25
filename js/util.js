function mod(a, b) {
    return ((a % b) + b) % b;
}

// Helper function to calculate the distance between two lat/lng points
function calculateDistance(lat1, lng1, lat2, lng2) {
    const R = 111320; // Approximate meters per degree of latitude

    const deltaY = (lat2 - lat1) * R;
    const deltaX = (lng2 - lng1) * R * Math.cos((lat1 + lat2) * Math.PI / 360); // Latitude midpoint adjustment

    return Math.sqrt(deltaX * deltaX + deltaY * deltaY);
}

function aStarRadius(start, end, graph, radius) {
    const openSet = new MinHeap();
    const gScore = {}; // Stores the cost from start to each node
    const fScore = {}; // Estimated cost from start to goal
    const cameFrom = {}; // Stores the previous node for each node

    const [startLat, startLng] = start.split(',').map(Number);


    // Initialize gScore and fScore for all nodes
    for (const node in graph) {
        gScore[node] = Infinity;
        fScore[node] = Infinity;
    }
    gScore[start] = 0;
    fScore[start] = heuristic(start, end);

    openSet.enqueue(start, fScore[start]);

    const path = [];

    while (!openSet.isEmpty()) {
        const current = openSet.dequeue();

        // If we have reached the goal, reconstruct the path
        if (current === end) {
            let temp = current;
            while (temp !== start) {
                path.push(temp);
                temp = cameFrom[temp];
            }
            path.push(start);
            return path.reverse();
        }

        // Check neighbors of the current node
        for (const neighbor of graph[current]) {
            const { node, weight, traveled } = neighbor;

            // Calculate the Euclidean distance from the start node
            const [nodeLat, nodeLng] = node.split(',').map(Number);
            const distanceFromStart = calculateDistance(startLat, startLng, nodeLat, nodeLng);

            // Skip neighbor if it's outside the radius
            if (distanceFromStart > radius) continue;

            // Apply penalty to traveled edges
            const edgeWeight = traveled ? weight * penalty : weight;

            // Calculate tentative g-score
            const tentativeGScore = gScore[current] + edgeWeight;

            // If this path is better, update scores and enqueue the neighbor
            if (tentativeGScore < gScore[node]) {
                cameFrom[node] = current;
                gScore[node] = tentativeGScore;
                fScore[node] = gScore[node] + heuristic(node, end);

                openSet.enqueue(node, fScore[node]);
            }
        }
    }

    // No path found
    return null;
}

function heuristic(node, end) {
    const [lat1, lng1] = node.split(',').map(Number);
    const [lat2, lng2] = end.split(',').map(Number);
    return calculateDistance(lat1, lng1, lat2, lng2)
}

// Function to get the nearest node to a specific lat/lng
function getNearestNode(lat, lng, graph) {
    let nearestNode = null;
    let minDistance = Infinity;

    // Iterate over all nodes in the graph and calculate the distance to the clicked position
    for (const node in graph) {
        const [nodeLat, nodeLng] = node.split(',').map(Number);
        const distance = calculateDistance(lat, lng, nodeLat, nodeLng);

        // Update nearest node if the distance is smaller
        if (distance < minDistance) {
            minDistance = distance;
            nearestNode = node;
        }
    }

    return nearestNode;
}

function saveAsJSFile(obj) {
    // Convert the graph object to a string
    const graphData = `${JSON.stringify(obj, null, 2)};`;
    
    // Create a Blob with the graph data and set it as a JavaScript file
    const blob = new Blob([graphData], { type: 'application/javascript' });
    
    // Create a link element to trigger the download
    const a = document.createElement('a');
    const url = URL.createObjectURL(blob);
    a.href = url;
    a.download = 'graph.js';  // Set the filename to graph.js
    a.click();  // Trigger the download
    
    // Revoke the object URL after the download
    URL.revokeObjectURL(url);
}

function coordinatesToGPX(coordinates, fileName = "route.gpx") {
    // Extract coordinates from the polyline
    // Start building the GPX file content
    let gpxContent = `<?xml version="1.0" encoding="UTF-8"?>\n`;
    gpxContent += `<gpx version="1.1" creator="Leaflet" xmlns="http://www.topografix.com/GPX/1/1">\n`;
    gpxContent += `  <trk>\n    <name>Exported Route</name>\n    <trkseg>\n`;

    // Add each coordinate as a GPX track point
    coordinates.forEach(point => {
        coord = point.split(",")
        gpxContent += `      <trkpt lat="${coord[0]}" lon="${coord[1]}"></trkpt>\n`;
    });

    // Close the GPX structure
    gpxContent += `    </trkseg>\n  </trk>\n</gpx>`;

    // Create a Blob for the GPX file
    const blob = new Blob([gpxContent], { type: "application/gpx+xml" });

    // Create a download link and trigger it
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}


function waitForDOMUpdate() {
    return new Promise(resolve => {
        setTimeout(resolve, 0);
    });
}

async function findEulerianCircuitWithVisualization(graph, map) {
    console.log(graph['51.60578,-0.05550'], "hawk")
    let circuit = [];
    let adjCopy = JSON.parse(JSON.stringify(graph)); // Deep copy of the graph
    let currentNode = Object.keys(adjCopy)[0]; // Start with an arbitrary node
    let usedEdges = new Set(); // Track used edges

    for (let i in graph){
        console.log(graph[i].length % 2 == 1)
    }

    // Helper function to parse "lat,lng" string into an object
    function parseLatLng(coordString) {
        const [lat, lng] = coordString.split(',').map(Number);
        return { lat, lng };
    }

    // Helper function to create a unique edge key (for undirected edges)
    function createEdgeKey(node1, node2) {
        return [node1, node2].sort().join("-");
    }

    // Helper function to delay execution
    function sleep(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    }

    // Helper function to visualize the traversal
    async function visualizeEdge(fromCoord, toCoord) {
        if (fromCoord == '51.60578,-0.05550')console.log(fromCoord, toCoord)
        
        const from = parseLatLng(fromCoord);
        const to = parseLatLng(toCoord);

        // Add a polyline to represent the edge being traversed
        L.polyline([from, to], { color: "orange", weight: 2 }).addTo(map);

        // Add markers at the start and end of the edge
        L.circleMarker([from.lat, from.lng], { color: "red" }).addTo(map);
        L.circleMarker([to.lat, to.lng], { color: "green" }).addTo(map);

        // Wait for a delay before proceeding
        await sleep(500);
    }

    // Recursive DFS function with visualization
    async function dfs(node) {
        while (adjCopy[node] && adjCopy[node].length > 0) {
            let edge = adjCopy[node].pop(); // Get the next edge
            let edgeKey = `${node}-${edge.node}`;  // Unique edge key
            let reverseEdgeKey = `${edge.node}-${node}`;  // For undirected graph
            console.log(edgeKey)

            // Check if this edge has already been used
            console.log()
            if (!usedEdges.has(edgeKey) && !usedEdges.has(reverseEdgeKey)) {
                usedEdges.add(edgeKey);  // Mark edge as used
                if(!edge.matched){
                    usedEdges.add(reverseEdgeKey);  // Mark reverse edge as used
                }else{
                    console.log("ayo", edgeKey)
                }
                await visualizeEdge(node, edge.node);
                await dfs(edge.node);
            }
        }
        circuit.push(node); // Add node to circuit after exploring all edges
    }

    await dfs(currentNode); // Start DFS traversal
    return circuit.reverse(); // Reverse the circuit since we add nodes post-traversal
}

function visualiseGraph(graph, filter = []){
    for (let node in graph){
        if (node == '51.60239,-0.06451')console.log(graph[node])
        if ((filter.length > 0 && filter.includes(node)) || filter.length == 0){
            console.log(graph[node].length)
            for (let to of graph[node]){
                if (node == '51.60239,-0.06451')console.log(to)
                convertToPolyline([node, to.node])
            }
        }
    }
}
