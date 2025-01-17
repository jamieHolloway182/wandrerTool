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

function calculateNodeDistance(node1, node2) {
    const [lat1, lng1] = node1.split(",").map(Number)
    const [lat2, lng2] = node2.split(",").map(Number)
    const R = 111320; // Approximate meters per degree of latitude

    const deltaY = (lat2 - lat1) * R;
    const deltaX = (lng2 - lng1) * R * Math.cos((lat1 + lat2) * Math.PI / 360); // Latitude midpoint adjustment

    return Math.sqrt(deltaX * deltaX + deltaY * deltaY);
}

function calculateManhattanDistance(key1, key2) {
    const [x1, y1] = key1.split(',').map(Number);
    const [x2, y2] = key2.split(',').map(Number);

    return Math.abs(x1 - x2) + Math.abs(y1 - y2);
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

function getNearestOddNode(start, graph, excluded=[]) {
    let nearestNode = null;
    let minDistance = Infinity;
    const [lat, lng] = start.split(',').map(Number);


    // Iterate over all nodes in the graph and calculate the distance to the clicked position
    for (const node in graph) {
        const [nodeLat, nodeLng] = node.split(',').map(Number);
        const distance = calculateDistance(lat, lng, nodeLat, nodeLng);

        // Update nearest node if the distance is smaller
        if (distance < minDistance && graph[node].length % 2 == 1 && !excluded.includes(node)) {
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


function waitForDOMUpdate(ms) {
    return new Promise(resolve => {
        setTimeout(resolve, ms);
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
        await sleep(5);
    }

    // Recursive DFS function with visualization
    async function dfs(node) {
        while (adjCopy[node] && adjCopy[node].length > 0 && adjCopy[node].find(n => !n.deadend)) {
            let edge = adjCopy[node].pop(); // Get the next edge
            let edgeKey = `${node}-${edge.node}`;  // Unique edge key
            let reverseEdgeKey = `${edge.node}-${node}`;  // For undirected graph
            console.log(edgeKey)

            // Check if this edge has already been used
            console.log()
            if (!usedEdges.has(edgeKey) && !usedEdges.has(reverseEdgeKey) && !edge.deadend) {
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
        console.log( adjCopy[node].length)
        for (let n = 0; n < adjCopy[node].length; n++){
            console.log(n)
            if (adjCopy[node][n].deadend){
                circuit.push(adjCopy[node][n].node)
                circuit.push(node)
                adjCopy[node].splice(n, 1)
            }
        }
    }

    await dfs(currentNode); // Start DFS traversal
    console.log("gay")
    console.log(circuit)
    return circuit.reverse(); // Reverse the circuit since we add nodes post-traversal
}

function visualiseGraph(graph, filter = []){
    for (let node in graph){
        if ((filter.length > 0 && filter.includes(node)) || filter.length == 0){
            for (let to of graph[node]){
                convertToPolyline([node, to.node]).addTo(map)
            }
        }
    }
}

function visualiseGraphWithMarker(graph, filter = []){
    for (let node in graph){
        if ((filter.length > 0 && filter.includes(node)) || filter.length == 0){
            L.marker(node.split(",")).addTo(map).bindPopup(node)
            for (let to of graph[node]){
                convertToPolyline([node, to.node]).addTo(map)
            }
        }
    }
}

function getGraphDistance(graph) {
    const visitedEdges = new Set(); // Track visited edges to prevent duplicates
    let totalDistance = 0;

    for (const [node, edges] of Object.entries(graph)) {
        for (const edge of edges) {
            if(!edge.virtual){

                const otherNode = edge.node; // Neighbor node
                const edgeKey = `${Math.min(node, otherNode)},${Math.max(node, otherNode)}`; // Canonical key for edge
                
                if (!visitedEdges.has(edgeKey)) {
                    visitedEdges.add(edgeKey);
                    
                    // Extract coordinates for both nodes
                    const [lat1, lng1] = node.split(',').map(Number);
                    const [lat2, lng2] = otherNode.split(',').map(Number);
                    
                    // Calculate the distance and add to the total
                    totalDistance += calculateDistance(lat1, lng1, lat2, lng2);
                }
            }
        }
    }

    return totalDistance;
}

function calculateGraphCenter(graph) {
    let minLat = Infinity;
    let maxLat = -Infinity;
    let minLng = Infinity;
    let maxLng = -Infinity;

    // Iterate over all nodes in the graph to find the min and max latitudes and longitudes
    for (const node in graph) {
        const [lat, lng] = node.split(',').map(Number);

        if (lat < minLat) minLat = lat;
        if (lat > maxLat) maxLat = lat;
        if (lng < minLng) minLng = lng;
        if (lng > maxLng) maxLng = lng;
    }

    // Calculate the center of the graph (the midpoint of the bounding box)
    const centerLat = (minLat + maxLat) / 2;
    const centerLng = (minLng + maxLng) / 2;

    return { centerLat, centerLng };
}

function calculateBBoxCenter(bbox) {
    const [topLeftLat, topLeftLng, bottomRightLat, bottomRightLng] = bbox;
    
    const centerLat = (topLeftLat + bottomRightLat) / 2;
    const centerLng = (topLeftLng + bottomRightLng) / 2;
    
    return {centerLat, centerLng};
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}


function convertToPolyline(circuit, color='green') {
    const polylinePoints = [];
    
    // Iterate through the circuit and add the coordinates of each node
    for (let i = 0; i < circuit.length ; i++) {
        let node = circuit[i];
        let coords = node.split(","); // Assuming the node is in the form "lat,lng"
        polylinePoints.push([parseFloat(coords[0]), parseFloat(coords[1])]);
    }
    
    // Create a polyline using Leaflet
    const polyline = L.polyline(polylinePoints, { color: color }).addTo(map);
    
    return polyline;
}

function splitGraphIntoCyclicAndAcyclic(graph) {
    let visited = new Set();
    let inCycle = new Set(); // Nodes in the cyclic component
    let cyclicComponent = {}; // The cyclic subgraph
    let acyclicSubgraphs = []; // Array of acyclic subgraphs

    // Helper to detect cycles using DFS
    function dfsCycleDetection(node, parent, path) {
        visited.add(node);
        path.add(node);

        for (let edge of graph[node]) {
            let neighbor = edge.node;

            // If the neighbor is in the current path, we've found a cycle
            if (path.has(neighbor)) {
                inCycle.add(node);
                inCycle.add(neighbor);
            } else if (!visited.has(neighbor)) {
                dfsCycleDetection(neighbor, node, path);
            }
        }

        path.delete(node); // Backtrack
    }

    // Step 1: Detect nodes in cycles
    for (let node in graph) {
        if (!visited.has(node)) {
            dfsCycleDetection(node, null, new Set());
        }
    }

    // Step 2: Build the cyclic component
    for (let node of inCycle) {
        cyclicComponent[node] = graph[node].filter(edge => inCycle.has(edge.node));
    }

    // Step 3: Extract acyclic subgraphs
    visited = new Set(inCycle); // Start fresh, but skip cyclic nodes
    function collectAcyclicSubgraph(startNode) {
        let subgraph = {};
        let stack = [startNode];
        visited.add(startNode);

        while (stack.length > 0) {
            let node = stack.pop();
            subgraph[node] = [];

            for (let edge of graph[node]) {
                subgraph[node].push(edge);
                if (!visited.has(edge.node) && !inCycle.has(edge.node)) {
                    stack.push(edge.node);
                    visited.add(edge.node);
                }
            }
        }

        return subgraph;
    }

    for (let node in graph) {
        if (!visited.has(node)) {
            acyclicSubgraphs.push(collectAcyclicSubgraph(node));
        }
    }

    return {
        cyclicComponent,
        acyclicSubgraphs
    };
}

function greedyMatchingHeuristic(oddNodes){
    const matched = new Set();
    const pairs = [];

    while (oddNodes.length > 1) {
        let minDist = Infinity;
        let pair = [];
        for (let i = 0; i < oddNodes.length; i++) {
            for (let j = i + 1; j < oddNodes.length; j++) {
                let nodeA = oddNodes[i];
                let nodeB = oddNodes[j];
                if (!matched.has(nodeA) && !matched.has(nodeB)) {
                    let distance = calculateManhattanDistance(nodeA, nodeB);
                    if (distance < minDist) {
                        minDist = distance;
                        pair = [nodeA, nodeB];
                    }
                }
            }
        }
        matched.add(pair[0]);
        matched.add(pair[1]);
        pairs.push(pair);
        oddNodes = oddNodes.filter(node => !matched.has(node)); // Remove paired nodes
    }

    return pairs;
}

function replaceVirtualEdgesWithRealPathsHeuristic(eulerianCircuit, subgraph, fullGraph, radius) {
    let realPath = [];
    console.log(eulerianCircuit, "hey")
    for (let i = 0; i < eulerianCircuit.length - 1; i++) {
        console.log(i, eulerianCircuit.length)
        const from = eulerianCircuit[i];
        const to = eulerianCircuit[i + 1];

        // Check if the edge is virtual
        const node = (subgraph[from].find(o => o.node == to));

        
        if (node){
            console.log("node")
            if (node.virtual) {
                console.log("virtual")
                // Replace virtual edge with real path using aStar
                var realSubPath = aStarRadius(from, to, fullGraph, radius);
                if (realSubPath) {
                    // Add the real path (omit the first node to avoid duplication)
                    realPath.push(...realSubPath.slice(0, realSubPath.length - 1));
                } else {
                    console.error("Failed to find real path for virtual edge: ${from} -> ${to}");
                    realPath.push(from)
                }
            } else {
                // Add direct edge for non-virtual connections
                console.log("real")
                realPath.push(from);
            }
        }else{
            console.log("no node", fullGraph[from].find(o => o.node == to))
            var realSubPath = aStarRadius(from, to, fullGraph, radius);
            if (realSubPath){
                realPath.push(...realSubPath.slice(0, realSubPath.length - 1));
            }else {
                console.error("Failed to find replacement path: ${from} -> ${to}");
                realPath.push(from)
            }
        }
    }

    // Add the last node to complete the circuit
    realPath.push(eulerianCircuit[eulerianCircuit.length - 1]);

    return realPath;
}

function dfsWithBacktracking(graph, startNode) {
    let visited = new Set();    // To keep track of visited nodes
    let order = [];             // To store the search order, including backtracking

    function dfs(node) {
        order.push(node);       // Visit the current node
        visited.add(node);

        for (let edge of graph[node] || []) {
            let neighbor = edge.node;
            if (!visited.has(neighbor)) {
                dfs(neighbor);  // Recursive call for unvisited neighbors
                order.push(node); // Backtracking to the current node
            }
        }
    }

    dfs(startNode);
    return order;
}

function dfsVisitEveryEdge(graph, startNode) {
    let visitedNodes = new Set(); // To keep track of visited nodes
    let visitedEdges = new Set(); // To track visited edges (undirected edges)
    let order = [];               // To store the order of visits, including edges

    function dfs(node) {
        visitedNodes.add(node);
        order.push(node);

        for (let edge of graph[node] || []) {
            let neighbor = edge.node;
            let edgeKey = node < neighbor ? `${node}-${neighbor}` : `${neighbor}-${node}`;

            if (!visitedEdges.has(edgeKey)) {
                // Mark the edge as visited
                visitedEdges.add(edgeKey);

                // Visit the neighbor
                dfs(neighbor);

                // Backtrack: Add the edge again for the reverse direction
                order.push(node);
            }
        }
    }

    dfs(startNode);
    return order;
}

function isPalindromeArray(arr) {
    let n = arr.length;
    for (let i = 0; i < n / 2; i++) {
        if (arr[i] !== arr[n - i - 1]) {
            return false; // If any mismatch is found, it's not a palindrome
        }
    }
    return true; // If no mismatches are found, it's a palindrome
}

function sliceFromToNext(arr, firstIndex) {
    let value = arr[firstIndex]
    let nextIndex = arr.indexOf(value, firstIndex + 1); // Find the next occurrence after the first

    // Return the slice if both occurrences exist; otherwise, return an empty array
    if (firstIndex !== -1 && nextIndex !== -1) {
        return arr.slice(firstIndex, nextIndex + 1);
    }
    return [];
}

function convertToPolyline(circuit, color = "green") {
    const polylinePoints = [];
    
    // Iterate through the circuit and add the coordinates of each node
    for (let i = 0; i < circuit.length ; i++) {
        let node = circuit[i];
        let coords = node.split(","); // Assuming the node is in the form "lat,lng"
        polylinePoints.push([parseFloat(coords[0]), parseFloat(coords[1])]);
    }
    
    // Create a polyline using Leaflet
    const polyline = L.polyline(polylinePoints, { color: color }).addTo(map);
    
    return polyline;
}

function convertToPolylineDelay(circuit, color = "green", ms=1) {
    const polylinePoints = [];
    
    // Iterate through the circuit and add the coordinates of each node
    for (let i = 0; i < circuit.length ; i++) {
        let node = circuit[i];
        let coords = node.split(","); // Assuming the node is in the form "lat,lng"
        polylinePoints.push([parseFloat(coords[0]), parseFloat(coords[1])]);
        waitForDOMUpdate(ms)
    }
    
    // Create a polyline using Leaflet
    const polyline = L.polyline(polylinePoints, { color: color }).addTo(map);
    
    return polyline;
}

function filterPath(path, graph) {
    // Helper function to count edges for a node
    function countEdges(node) {
        return graph[node] ? graph[node].length : 0;
    }

    return countEdges(path[0]) == 1 || countEdges(path[path.length - 1]) == 1;
}

function halvePalindrome(arr) {
    // Check if the array is a palindrome (optional)
    if (arr.join('') !== arr.reverse().join('')) {
        throw new Error("The array is not a palindrome");
    }

    // Return the first half of the array
    return arr.slice(0, Math.ceil(arr.length / 2));
}

function isCircle(graph) {
    return Object.keys(graph).every(node => graph[node].length === 2);
}

const calculateCentroid = (polygon) => {
    let xSum = 0, ySum = 0;
    polygon.forEach(([x, y]) => {
      xSum += x;
      ySum += y;
    });
    return [xSum / polygon.length, ySum / polygon.length];
  };

  function calculateAngle([cx, cy], [x, y]) {
    return Math.atan2(y - cy, x - cx);
  }

  // Calculate internal angle between three points (in degrees)
  function calculateAngleBetween(p1, p2, p3) {
    const a = Math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2);
    const b = Math.sqrt((p2[0] - p3[0]) ** 2 + (p2[1] - p3[1]) ** 2);
    const c = Math.sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2);
    const angle = Math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b));
    return (angle * 180) / Math.PI;
  }

  // Order points to minimize internal angles
  function orderPointsMinimizingAngles(points) {
    const centroid = calculateCentroid(points);

    // Sort points radially around the centroid
    points.sort((a, b) => calculateAngle(centroid, a) - calculateAngle(centroid, b));

    // Check and adjust for internal angles
    for (let i = 0; i < points.length; i++) {
      const prev = points[(i - 1 + points.length) % points.length];
      const current = points[i];
      const next = points[(i + 1) % points.length];

      const angle = calculateAngleBetween(prev, current, next);

      // If the angle is too sharp (e.g., < 10 degrees), flip order locally
      if (angle < 10) {
        [points[i], points[(i + 1) % points.length]] = [points[(i + 1) % points.length], points[i]];
      }
    }

    return points;
  }

  function orientation(p, q, r) {
    const val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1]);
    if (val === 0) return 0;  // collinear
    return (val > 0) ? 1 : 2;  // clockwise or counterclockwise
  }

  // Check if point q lies on segment pr
  function onSegment(p, q, r) {
    return (q[0] <= Math.max(p[0], r[0]) && q[0] >= Math.min(p[0], r[0]) &&
            q[1] <= Math.max(p[1], r[1]) && q[1] >= Math.min(p[1], r[1]));
  }

  // Check if two line segments (p1-p2) and (p3-p4) intersect
  function doIntersect(p1, p2, p3, p4) {
    const o1 = orientation(p1, p2, p3);
    const o2 = orientation(p1, p2, p4);
    const o3 = orientation(p3, p4, p1);
    const o4 = orientation(p3, p4, p2);

    if (o1 !== o2 && o3 !== o4) return true;
    if (o1 === 0 && onSegment(p1, p3, p2)) return true;
    if (o2 === 0 && onSegment(p1, p4, p2)) return true;
    if (o3 === 0 && onSegment(p3, p1, p4)) return true;
    if (o4 === 0 && onSegment(p3, p2, p4)) return true;

    return false;
  }

  // Check if a polygon has intersecting edges
  function hasIntersectingEdges(polygon) {
    const n = polygon.length;
    for (let i = 0; i < n; i++) {
      const p1 = polygon[i];
      const p2 = polygon[(i + 1) % n];  // Next point (wrap around)
      for (let j = i + 2; j < n; j++) {
        const p3 = polygon[j];
        const p4 = polygon[(j + 1) % n];  // Next point (wrap around)

        // Skip adjacent edges
        if (i === 0 && j === n - 1) continue;

        if (doIntersect(p1, p2, p3, p4)) {
          return true;  // If any pair of edges intersect
        }
      }
    }
    return false;
  }

  const calculateBoundingRadius = (polygon, centroid) => {
    const [cx, cy] = centroid;
    return Math.max(...polygon.map(([x, y]) => calculateDistance(cx, cy, x, y)));
  };

  const isPointInPolygon = (point, polygon) => {
    const [px, py] = point;
    let isInside = false;
  
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const [x1, y1] = polygon[i];
      const [x2, y2] = polygon[j];
  
      const intersects = ((y1 > py) !== (y2 > py)) &&
        (px < ((x2 - x1) * (py - y1)) / (y2 - y1) + x1);
  
      if (intersects) {
        isInside = !isInside;
      }
    }
  
    return isInside;
  };
  
  const isPointInPolygonWithRadius = (point, polygon) => {
    const centroid = calculateCentroid(polygon);
    const boundingRadius = calculateBoundingRadius(polygon, centroid);
    const [px, py] = point;
    const [cx, cy] = centroid;
  
    // Quick bounding radius check
    const distance = Math.sqrt((px - cx) ** 2 + (py - cy) ** 2);
    if (distance > boundingRadius) return false;
  
    // Perform full point-in-polygon test
    return isPointInPolygon(point, polygon);
  };