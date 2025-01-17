// Check if the point is inside the bounding box
function isPointInBBox(lat, lng, bbox) {
    const lat1 = bbox[0], lng1 = bbox[1], lat2 = bbox[2], lng2 = bbox[3];
    return lat <= lat1 && lat >= lat2 && lng >= lng1 && lng <= lng2;
}

// Create the subgraph with edges that go beyond the bounding box
function createSubgraph(graph, bbox, removeTraveled, radius) {
    let subgraph = {};
    
    // Filter out nodes in the bbox and initialize empty edges for them
    for (let node in graph) {

        // Check if node is within the bounding box
        if (isPointInPolygonWithRadius(node.split(",").map(Number), bbox, radius) && (graph[node].find(o => !o.traveled || !removeTraveled))) {
            subgraph[node] = [];  // Initialize empty edges for this node
        }
    }

    // Add edges between nodes within the bounding box
    for (let node in subgraph) {
        for (let edge of graph[node]) {
            // Ensure that both nodes connected by an edge are in the subgraph
            let targetNode = edge.node;
            if (subgraph[targetNode] && (!edge.traveled || !removeTraveled)) {
                subgraph[node].push({ node: targetNode, weight: edge.weight, virtual: false });
                // subgraph[targetNode].push({ node: node, weight: edge.weight, virtual: false });  // Add reverse edge
            }
        }
    }

    return subgraph;
}
            
function getDisconnectedComponents(subgraph) {
    let visited = new Set();
    let components = [];

    // Helper function to perform DFS traversal
    function dfs(node, component) {
        if (visited.has(node)) return;
        visited.add(node);
        component.push(node);
        
        if (subgraph[node]) {
            subgraph[node].forEach(edge => {
                dfs(edge.node, component);
            });
        }
    }
    
    // Find all disconnected components
    for (let node in subgraph) {
        if (!visited.has(node)) {
            let component = [];
            dfs(node, component);
            components.push(component);
        }
    }
    
    return components;
}

function removeFullyDisconnectedComponents(components, fullGraph, subgraph, radius) {
    const mainComponent = components.reduce((longest, current) => current.length > longest.length ? current : longest, []);

    const visited = new Set(); // Track visited nodes

    // Perform an iterative BFS/DFS to traverse the graph within the radius
    const stack = [mainComponent[0]]; // Start with the first node in the main component
    visited.add(mainComponent[0]);

    while (stack.length > 0) {
        const current = stack.pop(); // DFS (use queue.shift() for BFS)

        for (let neighbor of fullGraph[current] || []) {
            const [startLat, startLng] = mainComponent[0].split(',').map(parseFloat); // Start node coordinates
            const [lat, lng] = neighbor.node.split(',').map(parseFloat); // Neighbor coordinates

            const distance = calculateDistance(startLat, startLng, lat, lng); // Distance from the start node
            if (distance <= radius && !visited.has(neighbor.node)) {
                visited.add(neighbor.node);
                stack.push(neighbor.node);
            }
        }
    }

    // Filter components to retain only those with at least one visited node
    const newComponents = components.filter(component =>
        component.some(node => visited.has(node))
    );

    // Remove nodes of unvisited components from the subgraph
    for (let component of components) {
        if (!component.some(node => visited.has(node))) {
            component.forEach(node => delete subgraph[node]);
        }
    }

    return newComponents;
}

function connectComponentsWithCycle(subgraph, fullGraph, components) {
    // Step 1: Calculate distances between components
    let distances = [];
    for (let i = 0; i < components.length; i++) {
        for (let j = i + 1; j < components.length; j++) {
            let component1 = components[i];
            let component2 = components[j];
            
            let closestDistance = Infinity;
            let closestEdge = null;
            
            for (let node1 of component1) {
                for (let node2 of component2) {
                    let distance = calculateDistance(
                        parseFloat(node1.split(",")[0]),
                        parseFloat(node1.split(",")[1]),
                        parseFloat(node2.split(",")[0]),
                        parseFloat(node2.split(",")[1])
                    );
                    if (distance < closestDistance) {
                        closestDistance = distance;
                        closestEdge = { from: node1, to: node2, weight: closestDistance };
                    }
                }
            }
            
            distances.push({
                component1: i,
                component2: j,
                edge: closestEdge,
                weight: closestDistance
            });
        }
    }

    // Step 2: Approximate TSP to find a cycle
    let cycle = approximateTSP(components.length, distances);

    // Step 3: Add the connecting edges for the cycle
    for (let i = 0; i < cycle.length; i++) {
        let from = cycle[i];
        let to = cycle[(i + 1) % cycle.length]; // Wrap around to form a cycle

        let edge = distances.find(
            d => (d.component1 === from && d.component2 === to) ||
                 (d.component1 === to && d.component2 === from)
        ).edge;

        // Ensure nodes exist in subgraph
        if (!subgraph[edge.from]) subgraph[edge.from] = [];
        if (!subgraph[edge.to]) subgraph[edge.to] = [];
        
        // Add the edge to the subgraph
        subgraph[edge.from].push({ node: edge.to, weight: edge.weight, virtual: true });
        subgraph[edge.to].push({ node: edge.from, weight: edge.weight, virtual: true });
    }
}

// Example of an approximate TSP using Nearest Neighbor
function approximateTSP(numComponents, distances) {
    let visited = new Set();
    let cycle = [];
    let current = 0; // Start at component 0

    while (visited.size < numComponents) {
        cycle.push(current);
        visited.add(current);

        // Find the closest unvisited component
        let closest = null;
        let closestDistance = Infinity;

        for (let d of distances) {
            if (d.component1 === current && !visited.has(d.component2) && d.weight < closestDistance) {
                closest = d.component2;
                closestDistance = d.weight;
            } else if (d.component2 === current && !visited.has(d.component1) && d.weight < closestDistance) {
                closest = d.component1;
                closestDistance = d.weight;
            }
        }

        current = closest;
    }

    return cycle;
}

function improvedApproximatePath(components) {
    const visited = new Set();
    const path = [];
    const closestNodes = [];
    let current = components[0]; // Start at the first component in the list

    // Precompute distances between components
    const componentDistances = new Map();

    function calculateDistance(comp1, comp2) {
        if (componentDistances.has(`${comp1}-${comp2}`)) {
            return componentDistances.get(`${comp1}-${comp2}`);
        }

        let minDistance = Infinity;
        let closestPair = [];

        for (let key1 of comp1) {
            for (let key2 of comp2) {
                const dist = calculateNodeDistance(key1, key2);
                if (dist < minDistance) {
                    minDistance = dist;
                    closestPair = [key1, key2];
                }
            }
        }

        componentDistances.set(`${comp1}-${comp2}`, { distance: minDistance, pair: closestPair });
        componentDistances.set(`${comp2}-${comp1}`, { distance: minDistance, pair: closestPair }); // Symmetric
        return { distance: minDistance, pair: closestPair };
    }

    // Greedy initial solution
    while (visited.size < components.length) {
        path.push(current);
        visited.add(current);

        let closest = null;
        let closestDistance = Infinity;
        let closestPair = [];

        for (let next of components) {
            if (!visited.has(next)) {
                const { distance, pair } = calculateDistance(current, next);
                if (distance < closestDistance) {
                    closest = next;
                    closestDistance = distance;
                    closestPair = pair;
                }
            }
        }

        if (closest) {
            closestNodes.push(closestPair);
        }

        current = closest;
    }

    // Path refinement using Two-Opt
    function refinePath(path) {
        let improved = true;

        while (improved) {
            improved = false;
            for (let i = 1; i < path.length - 2; i++) {
                for (let j = i + 1; j < path.length - 1; j++) {
                    const currentDistance =
                        calculateDistance(path[i - 1], path[i]).distance +
                        calculateDistance(path[j], path[j + 1]).distance;

                    const swappedDistance =
                        calculateDistance(path[i - 1], path[j]).distance +
                        calculateDistance(path[i], path[j + 1]).distance;

                    if (swappedDistance < currentDistance) {
                        // Swap the segment
                        const reversedSegment = path.slice(i, j + 1).reverse();
                        path.splice(i, j - i + 1, ...reversedSegment);
                        improved = true;
                    }
                }
            }
        }

        return path;
    }

    const refinedPath = refinePath(path);

    return { path: refinedPath, closestNodes };
}

function approximatePathWithMST(components) {
    const numComponents = components.length;

    // Function to calculate the minimum distance between two components
    function calculateDistance(comp1, comp2) {
        let minDistance = Infinity;
        let closestPair = [];

        for (let key1 of comp1) {
            for (let key2 of comp2) {
                const dist = calculateNodeDistance(key1, key2);
                if (dist < minDistance) {
                    minDistance = dist;
                    closestPair = [key1, key2];
                }
            }
        }

        return { distance: minDistance, pair: closestPair };
    }

    // Step 1: Build the MST using Prim's algorithm
    const mstEdges = [];
    const visited = new Set();
    const edgeQueue = [];

    visited.add(0); // Start with the first component
    while (visited.size < numComponents) {
        for (let i of visited) {
            for (let j = 0; j < numComponents; j++) {
                if (!visited.has(j)) {
                    const { distance, pair } = calculateDistance(components[i], components[j]);
                    edgeQueue.push({ from: i, to: j, distance, pair });
                }
            }
        }

        // Find the smallest edge in the queue
        edgeQueue.sort((a, b) => a.distance - b.distance);
        const nextEdge = edgeQueue.shift();

        if (!visited.has(nextEdge.to)) {
            visited.add(nextEdge.to);
            mstEdges.push(nextEdge); // Add this edge to the MST
        }
    }

    // Step 2: Traverse the MST
    const adjacencyList = Array.from({ length: numComponents }, () => []);
    for (let edge of mstEdges) {
        adjacencyList[edge.from].push(edge.to);
        adjacencyList[edge.to].push(edge.from);
    }

    const path = [];
    const visitedNodes = new Set();
    const closestNodes = [];

    function dfs(node) {
        visitedNodes.add(node);
        path.push(node);

        for (let neighbor of adjacencyList[node]) {
            if (!visitedNodes.has(neighbor)) {
                closestNodes.push(
                    mstEdges.find(
                        (edge) => (edge.from === node && edge.to === neighbor) || (edge.from === neighbor && edge.to === node)
                    ).pair
                );
                dfs(neighbor);
            }
        }
    }

    dfs(0); // Start traversal at component 0

    // Step 3: Return the result
    return { path, closestNodes };
}

function approximatePath(components) {
    let visited = new Set();
    let path = [];
    let closestNodes = []; // 2D array to store closest nodes between successive components
    let currentIndex = 0; // Start at the first component in the list (index)

    // Helper function to find the closest pair of nodes between two components
    function findClosestNodes(comp1, comp2) {
        let minDistance = Infinity;
        let closestPair = [];

        for (let key1 of comp1) {
            for (let key2 of comp2) {
                const dist = calculateNodeDistance(key1, key2);
                if (dist < minDistance) {
                    minDistance = dist;
                    closestPair = [key1, key2];
                }
            }
        }

        return closestPair;
    }

    while (visited.size < components.length) {
        path.push(currentIndex);  // Add the current index to the path
        visited.add(currentIndex);

        let closest = null;
        let closestDistance = Infinity;
        let closestPair = [];

        for (let i = 0; i < components.length; i++) {
            if (!visited.has(i)) {
                const pair = findClosestNodes(components[currentIndex], components[i]);
                const distance = calculateNodeDistance(pair[0], pair[1]);

                if (distance < closestDistance) {
                    closest = i;
                    closestDistance = distance;
                    closestPair = pair;
                }
            }
        }

        if (closest !== null) {
            closestNodes.push(closestPair);
        }

        currentIndex = closest; // Move to the next closest component
    }

    return { path, closestNodes };
}

function floydWarshall(graph) {
    const nodes = Object.keys(graph); // List of node IDs
    const dist = {}; // Distance dictionary
    const next = {}; // Path reconstruction dictionary

    // Initialize distance and next dictionaries
    nodes.forEach((u) => {
        dist[u] = {};
        next[u] = {};
        nodes.forEach((v) => {
            if (u === v) {
                dist[u][v] = 0; // Distance to itself is 0
                next[u][v] = null; // No predecessor needed for self-loop
            } else {
                dist[u][v] = Infinity; // Initially set to Infinity
                next[u][v] = null; // No known path initially
            }
        });
    });

    // Set distances and predecessors for direct edges
    for (let u in graph) {
        for (let edge of graph[u]) {
            dist[u][edge.node] = edge.weight; // Weight of the edge from u to edge.node
            next[u][edge.node] = edge.node; // Direct edge, so the next node is the neighbor
        }
    }

    // Floyd-Warshall main algorithm
    for (let k of nodes) {
        for (let i of nodes) {
            for (let j of nodes) {
                if (dist[i][j] > dist[i][k] + dist[k][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                    next[i][j] = next[i][k]; // Update the path to go through k
                }
            }
        }
    }

    return { dist, next }; // Return both distance and path reconstruction matrices
}


// Greedy matching for odd degree nodes
function greedyMatching(oddNodes, dist, graph) {
    const matched = new Set();
    const pairs = [];

    while (oddNodes.length > 1) {
        let minDist = Infinity;
        let pair = [];
        
        // Iterate over all pairs of odd nodes
        console.log(oddNodes)
        for (let i = 0; i < oddNodes.length; i++) {
            console.log(i)
            for (let j = i + 1; j < oddNodes.length; j++) {
                let nodeA = oddNodes[i];
                let nodeB = oddNodes[j];

                // Check if nodes are already matched or are neighbors
                if (!matched.has(nodeA) && !matched.has(nodeB) && graph[nodeA].findIndex(e => e.node == nodeB) == -1) {
                    let distance = dist[nodeA][nodeB];
                    if (distance < minDist) {
                        minDist = distance;
                        pair = [nodeA, nodeB];
                    }
                }
            }
        }
        
        // Mark nodes as matched and store the pair
        matched.add(pair[0]);
        matched.add(pair[1]);
        pairs.push(pair);
        
        // Remove paired nodes from the oddNodes array
        oddNodes = oddNodes.filter(node => !matched.has(node));
    }

    return pairs;
}

// Add virtual edges based on the matching of odd-degree nodes
function addVirtualEdges(subgraph, matching) {
    for (let pair of matching) {
        subgraph[pair[0]].push({ node: pair[1], weight: 0, virtual: true });
        subgraph[pair[1]].push({ node: pair[0], weight: 0, virtual: true });
    }
}

function findEulerianPath(adjCopy, startNode) {

    let circuit = [];

    function dfs(node) {
        while (adjCopy[node] && adjCopy[node].length > 0) {
            let edge = adjCopy[node].pop(); // Get the next edge
            let neighbor = edge.node;

            adjCopy[neighbor] = [...adjCopy[neighbor].filter(e => e.node !== node)];

            dfs(neighbor); // Recursively visit the neighbor
        }
        circuit.push(node); // Add node to circuit after exploring all edges
    }

    dfs(startNode); // Start the DFS traversal from the given start node

    return circuit.reverse(); // Reverse the circuit since we add nodes post-traversal
}


function replaceVirtualEdgesWithRealPaths(eulerianCircuit, subgraph, fullGraph, radius, next) {
    let realPath = [];
    
    for (let i = 0; i < eulerianCircuit.length - 1; i++) {
        const from = eulerianCircuit[i];
        const to = eulerianCircuit[i + 1];

        // Check if the edge is virtual
        const node = (subgraph[from].find(o => o.node == to));

        
        if (node){
            if (node.virtual) {
                // Replace virtual edge with real path using aStar
                var realSubPath = reconstructPath(next, from, to)
                realSubPath = realSubPath || aStarRadius(from, to, fullGraph, radius);
                if (realSubPath) {
                    // Add the real path (omit the first node to avoid duplication)
                    realPath.push(...realSubPath.slice(0, realSubPath.length - 1));
                } else {
                    console.error("Failed to find real path for virtual edge: ${from} -> ${to}");
                    realPath.push(from)
                }
            } else {
                // Add direct edge for non-virtual connections
                realPath.push(from);
            }
        }else{
            var realSubPath = reconstructPath(next, from, to)
            realSubPath = realSubPath || aStarRadius(from, to, fullGraph, radius);
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

function drawBoundingBox(map, bbox) {
    for (let i = 0; i < bbox.length - 1; i++){
        L.polyline([bbox[i], bbox[i+1]]).addTo(map)
    }
}

function recursivelyRemovePalindromes(arr, graph, component) {
    function findPalindromes(arr) {
        let palindromes = [];
        let start = 0;

        while (start < arr.length) {
            let foundPalindrome = false;

            for (let end = start + 3; end <= arr.length; end++) {
                let slice = arr.slice(start, end);
                if (
                    isPalindromeArray(slice) &&
                    filterPath(halvePalindrome(slice), graph) // Check halved palindrome
                ) {
                    palindromes.push({ start, end, slice });
                    start = end - 1; // Move the `start` to the end of the found palindrome
                    foundPalindrome = true;
                    break; // Exit the inner loop to skip smaller overlapping palindromes
                }
            }

            if (!foundPalindrome) {
                start++; // Move to the next element if no palindrome was found
            }
        }

        return palindromes;
    }

    function isPalindromeArray(arr) {
        let n = arr.length;
        for (let i = 0; i < Math.floor(n / 2); i++) {
            if (arr[i] !== arr[n - i - 1]) {
                return false;
            }
        }
        return true;
    }

    function halvePalindrome(arr) {
        return arr.slice(0, Math.ceil(arr.length / 2));
    }

    function filterPath(path, graph) {
        function countEdges(node) {
            return graph[node] ? graph[node].length : 0;
        }
        return countEdges(path[0]) === 1 || countEdges(path[path.length - 1]) === 1;
    }

    function removeNodesFromGraph(graph, nodesToRemove) {
        for (let node in graph) {
            graph[node] = graph[node].filter(edge => !nodesToRemove.includes(edge.node));
        }
        for (let node of nodesToRemove) {
            delete graph[node];
        }
    }

    let result = [...arr];
    let removedPalindromes = [];
    let palindromes = findPalindromes(result);

    while (palindromes.length > 0) {
        palindromes.sort((a, b) => b.slice.length - a.slice.length); // Sort by length (descending)
        let toProcess = palindromes.pop(); // Process the largest palindrome

        // Record the full palindrome and its middle part
        let middle = toProcess.slice.slice(1, toProcess.slice.length - 1); // Exclude start and end
        removedPalindromes.push({
            fullPalindrome: toProcess.slice,
            middlePart: middle
        });

        // Remove the middle part of the palindrome from `result`
        result.splice(
            toProcess.start + 1, // Start after the first element
            toProcess.end - toProcess.start - 1 // Length of the middle part
        );

        // Remove nodes in the middle part from the graph
        removeNodesFromGraph(graph, middle);

        // Adjust the indices of the remaining palindromes
        let shift = -(toProcess.end - toProcess.start - 1);
        palindromes = palindromes.map(p => {
            if (p.start > toProcess.start) {
                return { ...p, start: p.start + shift, end: p.end + shift };
            }
            return p;
        });

        // Find new palindromes after modification
        if (palindromes.length === 0) {
            palindromes = findPalindromes(result);
        }
    }

    graphCopy = graph;

    if (graphCopy[result[0]] && graphCopy[result[0]].length === 1) {
        let num = 1;
        while (graphCopy[result[num]].length === 2) {
            num++;
        }
        let path = result.slice(0, num + 1);

        let palindrome = [...path.slice().reverse().slice(0, path.length - 1), ...path];
        let middle = palindrome.slice(1, palindrome.length - 1); // Exclude start and end

        removedPalindromes.push({
            fullPalindrome: palindrome,
            middlePart: middle
        });

        result = result.slice(num, -num);
        removeNodesFromGraph(graph, middle);
    }

    for (let node in graphCopy) {
        if (!component.includes(node)) {
            delete graphCopy[node];
        }
    }

    return { result, removedPalindromes, graphCopy };
}


function isDeadendComponent(component, palindromes){
    let sum = 0;
    let flag = false

    for (let node of component ){
        flag = false
        for (let palindrome of palindromes){
            if(flag){
                break;
            }
            for (let pNode of palindrome.fullPalindrome){
                if (pNode == node){
                    sum++
                    flag = true
                    break;
                }
            }
        }
    }
    return sum >= component.length
}

function findCirclePath(graph, startNode, endNode) {
    // Traverse the entire circle to construct the full path
    const visited = new Set();
    let fullCirclePath = [];
    const fullCircleWeights = [];
    let currentNode = startNode;
    let previousNode = null;

    // Traverse the circle (clockwise or counterclockwise doesn't matter initially)
    do {
        visited.add(currentNode);
        fullCirclePath.push(currentNode);

        // Get neighbors and choose the next node (exclude the previous node to avoid backtracking)
        const neighbors = graph[currentNode];
        const nextEdge = neighbors.find(edge => edge.node !== previousNode);
        if (!nextEdge) break;

        fullCircleWeights.push(nextEdge.weight);
        previousNode = currentNode;
        currentNode = nextEdge.node;
    } while (currentNode !== startNode);

    // Find the indices of start and end nodes in the circle path
    const startIndex = fullCirclePath.indexOf(startNode);
    const endIndex = fullCirclePath.indexOf(endNode);

    // Calculate the total weight for each possible path
    const calculatePathWeight = (pathIndices) => {
        return pathIndices.reduce((sum, index, i) => {
            if (i < pathIndices.length - 1) {
                const from = fullCirclePath[index];
                const to = fullCirclePath[pathIndices[i + 1]];
                const edge = graph[from].find(e => e.node === to);
                return sum + edge.weight;
            }
            return sum;
        }, 0);
    };

    // Path 1: Clockwise (startIndex -> endIndex)
    const path1Indices = [];
    for (let i = startIndex; i <= endIndex; i++) path1Indices.push(i);
    const path1Weight = calculatePathWeight(path1Indices);

    // Path 2: Counterclockwise (endIndex -> startIndex, wrapping around)
    const path2Indices = [];
    for (let i = endIndex; i !== startIndex; i = (i + 1) % fullCirclePath.length) {
        path2Indices.push(i);
    }
    path2Indices.push(startIndex); // Include the startIndex at the end
    const path2Weight = calculatePathWeight(path2Indices);

    // Determine the shortest path
    let shortestPath =
        path1Weight <= path2Weight
            ? fullCirclePath.slice(startIndex, endIndex + 1)
            : [
                  ...fullCirclePath.slice(endIndex),
                  ...fullCirclePath.slice(0, startIndex + 1)
              ];

    fullCirclePath = [...fullCirclePath, ...shortestPath]
    return fullCirclePath;
}

function createTree(graph, start) {
    let tree = {};
    let visited = new Set();

    function dfs(node, parent, level) {
        visited.add(node);

        tree[node] = {
            parent: parent,
            children: [],
            level: level // Set the level as the distance from the start
        };

        for (let edge of graph[node]) {
            let neighbor = edge.node;

            if (!visited.has(neighbor)) {
                tree[node].children.push(neighbor);

                dfs(neighbor, node, level + 1); // Increment level for child nodes
            }
        }
    }

    dfs(start, null, 0); // Start at level 0

    return tree;
}

function findTreePath(tree, end){
    let current = tree[end]
    let path = [end]
    while(current.parent != null){
        path.push(current.parent)
        current = tree[current.parent]
    }
    return path.reverse()
}

function traverseTreeToEnd(tree, path) {
    let start = path[0]; // Starting node from path
    let end = path[path.length - 1]; // Ending node from path
    let current = start;

    let order = [start];

    while (current !== end) {
        // Check if there are any unexplored children excluding the path
        let children = [...tree[current].children].filter(c => !path.includes(c));

        if (children.length > 0) {
            // Explore the first child
            let child = children[0];
            tree[current].children.splice(tree[current].children.indexOf(child), 1); // Remove from the tree to mark as explored
            current = child;
            order.push(current);
        } else {
            // Backtrack only if there's an unexplored node with a lower level
            let shouldBacktrack = false;
            for (let node in tree) {
                if (tree[node].children.length > 0 && tree[node].level < tree[current].level) {
                    shouldBacktrack = true;
                    break;
                }
            }

            if (shouldBacktrack) {
                current = tree[current].parent;
                order.push(current);
            } else {
                // If no unexplored nodes with lower levels, proceed with the path
                let pathChild = tree[current].children[0];
                tree[current].children.splice(tree[current].children.indexOf(pathChild), 1); // Remove from the tree to mark as explored
                current = pathChild;
                order.push(current);
            }
        }
    }

    return order;
}

function CPP(subgraph, start, end, radius) {

    // Time the Floyd-Warshall step
    console.time("Floyd Warshall");
    const {dist, next} = floydWarshall(subgraph);
    console.timeEnd("Floyd Warshall");

    let oddNodes = [];
    for (let node in subgraph) {
        if (subgraph[node].length % 2 === 1 && node != start && node != end) {
            oddNodes.push(node);
        }
    }

    if (oddNodes.length > 0) {
        // Time the greedy matching step
        console.time("Greedy Matching");
        console.log("Greedy Matching");
        let matching = greedyMatching(oddNodes, dist, subgraph);
        console.timeEnd("Greedy Matching");

        // Time the add virtual edges step
        console.time("Add Virtual Edges");
        addVirtualEdges(subgraph, matching);
        console.timeEnd("Add Virtual Edges");
    }

    // Time the Eulerian circuit finding step
    console.time("Eulerian Circuit");
    console.log("Eulerian Circuit");
    let circuit = findEulerianPath(structuredClone(subgraph), start);
    console.timeEnd("Eulerian Circuit");

    // Time the replace virtual edges with real paths step
    console.time("Replace Virtual Edges");
    console.log("Replace Virtual Edges");
    circuit = replaceVirtualEdgesWithRealPaths(circuit, subgraph, graph, radius, next);
    console.timeEnd("Replace Virtual Edges");

    return circuit;
}
    
// Main function to solve the CPP on the subgraph
function solveCPP(bbox) {
    console.log("Starting CPP");

    console.time("Overall Time");

    const centroid = calculateCentroid(bbox)
    const radius = calculateBoundingRadius(bbox, centroid) * 1.5;
    console.log(radius)

    console.time("Subgraph Creation");
    let subgraph = createSubgraph(graph, bbox, true, radius);
    console.timeEnd("Subgraph Creation");

    console.time("Find Components");
    let components = getDisconnectedComponents(subgraph);
    console.log(components)
    components = components.filter(c => c.length > 1);
    components = removeFullyDisconnectedComponents(components, graph, subgraph, radius);
    console.log(components)
    console.timeEnd("Find Components");

    console.time("Approximate Path");
    let { path, closestNodes } = approximatePath(components);
    console.timeEnd("Approximate Path");

    console.log(path, closestNodes);
    let overallRoute = [];
    let componentsRoute = [];
    let paths = []

    for (let i = 0; i < components.length; i++) {
        console.log(`Processing Component ${i}`);
        console.time(`Component ${i} Time`);

        let c = components[path[i]];

        console.time("DFS Visit Every Edge");
        let search = dfsVisitEveryEdge(structuredClone(subgraph), c[0]);
        console.timeEnd("DFS Visit Every Edge");

        console.time("Remove Palindromes");
        var { result, removedPalindromes, graphCopy } = recursivelyRemovePalindromes(search, structuredClone(subgraph), c);
        console.timeEnd("Remove Palindromes");

        console.log("Removed Palindromes:", removedPalindromes);

        let deadendComponent = isDeadendComponent(c, removedPalindromes);

        let start, end, entrancePath, exitPath;


        if (i == 0) {
            start = c[0];
        } else {
            start = closestNodes[i - 1][1];
            entrancePath = paths[i - 1]
        }

        if (i == components.length - 1) {
            end = c[0];
        } else {
            end = closestNodes[i][0];
        }
        
        if (deadendComponent) {
            
            if (i > 0) {
                for (let n of entrancePath) {
                    if (c.includes(n)) {
                        start = n;
                        const index = entrancePath.indexOf(n);
                        entrancePath = entrancePath.slice(0, index);
                        paths[i - 1] = entrancePath
                        break;
                    }
                }
            }

            if (i != components.length - 1) {
                console.time("Exit Path");
                let nextStart = closestNodes[i][1];
                exitPath = aStarRadius(end, nextStart, graph, radius);
                console.timeEnd("Exit Path");
                for (let n of exitPath.reverse()) {
                    if (c.includes(n)) {
                        end = n;
                        const index = exitPath.indexOf(n);
                        exitPath = exitPath.slice(0, index + 1);
                        break;
                    }
                }
                paths.push(exitPath.reverse())
            }

            let route;
            if (start === end) {
                route = dfsVisitEveryEdge(structuredClone(subgraph), start);
            } else {
                let tree = createTree(subgraph, start);
                let path = findTreePath(tree, end);
                route = traverseTreeToEnd(tree, path);
            }
            componentsRoute.push(route);
        } else {
            console.log("Processing Circuit");
            console.time("Process Circuit");
            let circuit;
            if (isCircle(graphCopy)) {
                start = getNearestNode(...start.split(",").map(Number), graphCopy);
                end = getNearestNode(...end.split(",").map(Number), graphCopy);
                circuit = findCirclePath(structuredClone(graphCopy), start, end);
            } else {
                start = getNearestOddNode(start, graphCopy);
                end = getNearestOddNode(end, graphCopy, [start]);
                circuit = CPP(graphCopy, start, end);
            }
            while (removedPalindromes.length > 0) {
                let flag = false;
                for (let i = 0; i < removedPalindromes.length; i++) {
                    let index = circuit.findIndex(n => n == removedPalindromes[i].fullPalindrome[0]);
                    if (index != -1) {
                        flag = true;
                        let toEnter = removedPalindromes[i].fullPalindrome.slice(1);
                        circuit.splice(index + 1, 0, ...toEnter);
                        removedPalindromes.splice(i, 1);
                    }
                }
                if (!flag) {
                    console.log(circuit, removedPalindromes);
                    break;
                }
            }
            componentsRoute.push(circuit);

            let nextStart = closestNodes[i][1];
            exitPath = aStarRadius(end, nextStart, graph, radius);
            paths.push(exitPath)

            let previousComp = componentsRoute[i - 1]
            let previousEnd = previousComp[previousComp.length - 1]
            entrancePath = aStarRadius(previousEnd, start, graph, radius)
            paths[i - 1] = entrancePath

            console.timeEnd("Process Circuit");
        }

        console.timeEnd(`Component ${i} Time`);
    }

    console.time("Combine Routes");
    console.log(paths)
    paths.push([])
    for (let i = 0; i < componentsRoute.length; i++) {
        overallRoute = [...overallRoute, ...componentsRoute[i], ...paths[i]];
    }
    console.timeEnd("Combine Routes");

    console.timeEnd("Overall Time");

    return overallRoute;
}

function reconstructPath(next, start, end) {
    if (next[start][end] === null) {
        return null; // No path exists
    }

    const path = [start];
    let current = start;

    while (current !== end) {
        current = next[current][end];
        if (current === null) return null; // Path breaks, return null
        path.push(current);
    }

    return path;
}

