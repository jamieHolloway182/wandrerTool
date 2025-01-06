var text = document.getElementById("text")

map.on('click', function (e) {
    if (clicking){
        if (clickCount === 0) {
            route = []
            // Store the first click (start point)
            startClick = e.latlng;
            var startMarker = L.marker([startClick.lat, startClick.lng]).addTo(map);
            if (mode == "path"){
                startMarker.bindPopup('Start').openPopup();
                text.innerHTML = "Place end position"
            }

            clickCount++;
        } else if (clickCount === 1) {
            // Store the second click (goal point)
            goalClick = e.latlng;
            var endMarker = L.marker([goalClick.lat, goalClick.lng]).addTo(map)
            
            // Get the nearest nodes to the start and goal
            
            if (mode == "path"){

                text.innerHTML = "Processing..."
                waitForDOMUpdate(0).then(r => {

                    endMarker.bindPopup('Goal').openPopup();
                    const startNode = getNearestNode(startClick.lat, startClick.lng, graph);
                    const goalNode = getNearestNode(goalClick.lat, goalClick.lng, graph);
                    route = [startNode, goalNode]
                    
                    // Perform A* pathfinding
                    var path;
                    path = aStar(startNode, goalNode, graph)
                    path = path == null ? [] : path
    
                    coordinates = path
                    // Convert the path into a Leaflet polyline and add it to the map
                    text.innerHTML = "Press add button to start"
                    finish()
                })

            }else{
                bbox = createBoundingBox(startClick.lat, startClick.lng, goalClick.lat, goalClick.lng)
                drawBoundingBox(map, bbox)
                waitForDOMUpdate(1000).then(r => {
                    coordinates = solveCPP(bbox)
                    finish()
                })
            }

           
        }
    }
});

function finish(){
    var polyline = convertToPolyline(coordinates)
    var bounds = polyline.getBounds();
    map.fitBounds(bounds)
    
    // Reset the click count for the next pair of clicks
    clickCount = 0;
    startClick = null;
    goalClick = null;
}

var polylineGroup = L.featureGroup().addTo(map);

var penalty = 2

var route = []
var coordinates = []